import tensorflow as tf
from tensorflow.python.layers.core import Dense
from easydict import EasyDict
import logging
from tqdm import tqdm
import editdistance as ed
import numpy as np


class E2EModel:
    def __init__(self, config, abc, session):
        self.session = session
        self._logger = logging.getLogger('model')
        self._config = config
        self._abc = abc
        # inputs
        self._logger.info(f'Batch size is {self._config.batch_size}')
        self.sound_in = tf.placeholder(shape=[self._config.batch_size, None, self._config.features_num],
                                       dtype=tf.float32)
        self.sound_len = tf.placeholder(shape=[self._config.batch_size], dtype=tf.int32)
        self.text_target = tf.placeholder(shape=[self._config.batch_size, None],
                                          dtype=tf.int32)
        self.text_len = tf.placeholder(shape=[self._config.batch_size], dtype=tf.int32)
        self.seq_mask = tf.sequence_mask(self.text_len - 1, dtype=tf.float32)
        self.dropout_in = tf.placeholder(dtype=tf.float32)
        self.dropout_out = tf.placeholder(dtype=tf.float32)
        # inputs and targets preparation
        with tf.variable_scope('preprocessing'):
            self._hidden_repr, self._updated_len = self.input_preprocessing(self.sound_in, self.sound_len)
        self._logger.info(f'Decoder embedding size is {self._config.decoder_emb_size}')
        self.embedding_decoder = tf.get_variable('embedding_decoder',
                                                 [len(self._abc.vocab), self._config.decoder_emb_size])
        self.decoder_emb = tf.nn.embedding_lookup(self.embedding_decoder, self.text_target[:, :-1])
        self.projection_layer = Dense(len(self._abc.vocab), use_bias=False)
        # encoder-decoder
        self._logger.info('Preparing encoder graph...')
        with tf.variable_scope('encoder'):
            self.encoder = self.build_encoder(self._hidden_repr, self._updated_len)
        self._logger.info('Preparing decoder graph...')
        with tf.variable_scope('decoder'):
            self.decoder = self.build_decoder(self.decoder_emb, self.text_len - 1, self.projection_layer,
                                              self.embedding_decoder, self.encoder)
        # training part
        self._logger.info('Preparing optimizer graph...')
        self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.decoder.train.out[0],
                                                     targets=self.text_target[:, 1:],
                                                     weights=self.seq_mask)
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        optimizer = tf.train.AdamOptimizer(self._config.learn_rate)
        grads_and_vars = [(tf.clip_by_norm(g, self._config.clip_norm), v)
                          for g, v in optimizer.compute_gradients(self.loss)]
        self.train_op = optimizer.apply_gradients(grads_and_vars, self.global_step, name='apply_grads')
        self.total_params = sum([x.get_shape().num_elements() for x in tf.trainable_variables()])
        self._saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)
        self._logger.info(f'Model graph with {self.total_params/1e6:2.1f}M params ready.')

    def input_preprocessing(self, inputs, inputs_len):
        if self._config.conv_units <= 0:
            return inputs, inputs_len
        inputs_conv1 = tf.contrib.layers.convolution2d(inputs=tf.expand_dims(inputs, axis=-1),
                                                       num_outputs=self._config.conv_units, kernel_size=[3, 3],
                                                       stride=[1, 1])
        pooled = tf.nn.max_pool(inputs_conv1, [1, self._config.pool_step, 1, 1],
                                [1, self._config.pool_step, 1, 1], 'SAME')
        updated_len = tf.to_int32(tf.ceil(inputs_len / self._config.pool_step))
        shape = [int(x) if x.value is not None else -1 for x in pooled.get_shape()]
        shape[-2] *= shape[-1]
        shape = shape[0:-1]
        inputs_reshaped = tf.reshape(pooled, shape=shape, name="merge_last_dim")
        return inputs_reshaped, updated_len

    def _single_cell(self, units):
        cells = tf.contrib.rnn.LSTMCell(units)
        cells_drop = tf.nn.rnn_cell.DropoutWrapper(cell=cells, input_keep_prob=(1.0 - self.dropout_in),
                                                   output_keep_prob=(1.0 - self.dropout_out))
        return cells_drop

    def _bi_cell(self, units):
        with tf.variable_scope('forward'):
            fw_cell = self._single_cell(units)
        with tf.variable_scope('backward'):
            bw_cell = self._single_cell(units)
        return fw_cell, bw_cell

    def _make_cells(self, layers, units):
        self._logger.info(f'LSTM stacks: {layers} layers x {units} units.')
        cells = [self._single_cell(units) for _ in range(layers)]
        cells_enc = tf.nn.rnn_cell.MultiRNNCell(cells)
        return cells_enc

    def _encoder_bilayer(self, encoder_input, encoder_input_len):
        fw_cell, bw_cell = self._bi_cell(self._config.cell_units)
        encoder_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(
            fw_cell, bw_cell, encoder_input, sequence_length=encoder_input_len,
            time_major=False, dtype=tf.float32)
        return self._bilayer_to_flat(encoder_outputs, encoder_state)

    def _bilayer_to_flat(self, encoder_outputs, encoder_state):
        outputs = tf.concat(encoder_outputs, axis=-1, name='encoder-out')
        self._logger.info(f'Merging bi-layer into a flat one '
                          f'2x{encoder_outputs[0].get_shape()[-1]} -> {outputs.get_shape()[-1]}')
        concat_state = tf.contrib.rnn.LSTMStateTuple(
            c=tf.concat([encoder_state[0].c, encoder_state[1].c], axis=-1),
            h=tf.concat([encoder_state[0].h, encoder_state[1].h], axis=-1))
        self._logger.info(f'State sizes: c-{concat_state.c.get_shape()[-1]}, h-{concat_state.h.get_shape()[-1]}.')
        return outputs, concat_state

    def _pyramidal_stack(self, outputs, sequence_length):
        max_time = tf.shape(outputs)[1]
        num_units = outputs.get_shape().as_list()[-1]
        paddings = [[0, 0], [0, tf.floormod(max_time, 2)], [0, 0]]
        outputs = tf.pad(outputs, paddings)
        concat_outputs = tf.reshape(outputs, (self._config.batch_size, -1, num_units * 2))
        return concat_outputs, tf.floordiv(sequence_length, 2) + tf.floormod(sequence_length, 2)

    def build_encoder(self, encoder_input, encoder_input_len):
        states = []
        for layer in range(self._config.encoder_layers):
            encoder_input, encoder_input_len = self._pyramidal_stack(encoder_input, encoder_input_len)
            with tf.variable_scope('pyr_stack_{}'.format(layer)):
                encoder_input, state = self._encoder_bilayer(encoder_input, encoder_input_len)
            states.append(state)

        encoder = EasyDict()
        encoder.out = encoder_input
        encoder.state = tuple(states)
        encoder.len = encoder_input_len
        return encoder

    def build_decoder(self, decoder_inputs, decoder_inputs_len, output_layer, embedding_decoder,
                      encoder):
        cell_dec = self._make_cells(self._config.decoder_layers, encoder.state[0].c.get_shape()[-1])
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(encoder.out.get_shape()[-1], encoder.out,
                                                                   normalize=True)
        att_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell_dec, attention_mechanism,
                                                               attention_layer_size=self._config.attention_units,
                                                               alignment_history=True)
        decoder_initial_state = att_decoder_cell.zero_state(self._config.batch_size, tf.float32)
        if self._config.pass_state:
            # we assume that encoder is at least as deep as decoder
            decoder_initial_state = decoder_initial_state.clone(
                cell_state=tuple(encoder.state[self._config.encoder_layers - self._config.decoder_layers:]))
        if self._config.sampling_prob > 0 and embedding_decoder is not None:
            self._logger.info(f'Using {100 * self._config.sampling_prob:2.0f}% sampling probability for training.')
            helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                decoder_inputs, decoder_inputs_len, embedding_decoder,
                self._config.sampling_prob, time_major=False)
        else:
            self._logger.info('Running training in next symbol mode.')
            helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs, decoder_inputs_len, time_major=False)
        decoder = tf.contrib.seq2seq.BasicDecoder(att_decoder_cell, helper, decoder_initial_state,
                                                  output_layer=output_layer)
        output, state, output_length = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=False)
        decoder = EasyDict()
        decoder.train = EasyDict()
        decoder.train.out = output
        decoder.train.state = state
        decoder.train.len = output_length
        decoder.train.attention = state[-2].stack()
        if embedding_decoder is not None:
            self._logger.info('Using GreedyEmbeddingHelper for inference time decoding.')
            emb_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding_decoder,
                                                                  tf.fill([self._config.batch_size], 0), 0)
            emb_decoder = tf.contrib.seq2seq.BasicDecoder(att_decoder_cell, emb_helper, decoder_initial_state,
                                                          output_layer=output_layer)
            emb_outputs, emb_state, emb_length = tf.contrib.seq2seq.dynamic_decode(
                emb_decoder, maximum_iterations=self._config.max_iter, output_time_major=False)
            decoder.infer = EasyDict()
            decoder.infer.out = emb_outputs
            decoder.infer.state = emb_state
            decoder.infer.len = emb_length
            decoder.infer.attention = emb_state[-2].stack()
        return decoder

    def train(self, iterable):
        total_loss = 0
        handle = tqdm(iterable, unit='batch', desc='Training')
        for b in handle:
            feed_dict = {self.sound_in: b[0], self.sound_len: b[1], self.text_target: b[2], self.text_len: b[3],
                         self.dropout_in: self._config.dropout_in, self.dropout_out: self._config.dropout_out}
            _, loss, step = self.session.run([self.train_op, self.loss, self.global_step], feed_dict)
            total_loss = 0.9 * total_loss + 0.1 * loss
            handle.set_description(f'Train loss: {total_loss:2.2f}, step {step}')

    def get_metrics(self, iterable):
        total_loss = 0
        infer_distance = 0
        total_chars = 0
        batches_num = 0
        ideally = 0
        for b in tqdm(iterable, unit='batch', desc='Computing metrics'):
            batches_num += 1
            feed_dict = {self.sound_in: b[0], self.sound_len: b[1], self.text_target: b[2], self.text_len: b[3],
                         self.dropout_in: 0.0, self.dropout_out: 0.0}
            loss, trans_infer, len_infer = self.session.run(
                [self.loss, self.decoder.infer.out, self.decoder.infer.len], feed_dict)
            total_loss += loss
            for bi in range(self._config.batch_size):
                target_text = b[2][bi, 1:b[3][bi]].astype(np.int32)
                infer_text = trans_infer[1][bi, :len_infer[bi]]
                new_infer_dist = ed.eval(target_text, infer_text)
                infer_distance += new_infer_dist
                total_chars += len(target_text)
                if new_infer_dist < 1:
                    ideally += 1.0
        gt_text = ' '.join([self._abc.get_char(x) for x in target_text])
        in_text = ' '.join([self._abc.get_char(x) for x in infer_text])
        self._logger.info(f'Ground truth: {gt_text}')
        self._logger.info(f'Infer text  : {in_text}')
        total_loss /= batches_num
        infer_distance /= total_chars
        ideally /= batches_num * self._config.batch_size
        return total_loss, infer_distance, ideally

    def transcribe(self, features):
        batch_features = np.concatenate([features[np.newaxis, :]] * self._config.batch_size, axis=0)
        batch_len = [features.shape[0]] * self._config.batch_size
        trans, lens, align = self.session.run(
            [self.decoder.infer.out, self.decoder.infer.len, self.decoder.infer.attention],
            {self.sound_in: batch_features, self.sound_len: batch_len,
             self.dropout_in: 0.0, self.dropout_out: 0.0})
        char_ids = trans[1][0, :lens[0]]
        probs_mat = trans[0][0, :, :]
        return [self._abc.get_char(x) for x in char_ids], probs_mat, np.squeeze(align[:, 0, :])

    def save(self):
        self._saver.save(self.session, 'data/model')

    def load(self):
        try:
            self._saver.restore(self.session, 'data/model')
            self._logger.info('Model restored.')
        except ValueError:
            self._logger.error('Failed to restore model.')
