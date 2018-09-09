import nltk
import random
import logging
import numpy as np
import pickle


class CMUDataSource:
    def __init__(self, abc_gr, abc_ph):
        self.abc_gr = abc_gr
        self.abc_ph = abc_ph
        self._logger = logging.getLogger(__name__)
        random.seed(1224)
        data = list(nltk.corpus.cmudict.entries())
        random.shuffle(data)
        split_point = int(0.7 * len(data))
        self.train_data = sorted(data[:split_point], key=lambda x: len(x))
        self.train_data = [x for x in self.train_data if x[0].isalnum()]
        self.test_data = data[split_point:]
        self.test_data = [x for x in self.test_data if x[0].isalnum()]
        self._logger.info(f'Data split: {len(self.train_data)} train, {len(self.test_data)} test.')

    def save(self):
        with open('data/vocab_gr.pickle', 'wb') as f:
            pickle.dump(self.abc_gr, f)
        with open('data/vocab_ph.pickle', 'wb') as f:
            pickle.dump(self.abc_ph, f)

    def load(self):
        try:
            with open('data/vocab_gr.pickle', 'rb') as f:
                self.abc_gr = pickle.load(f)
            with open('data/vocab_ph.pickle', 'rb') as f:
                self.abc_ph = pickle.load(f)
            return True
        except FileNotFoundError:
            return False

    def compile_abc(self):
        for c, p in self.train_data:
            self.abc_gr.update(c)
            self.abc_ph.update(p)
        self.abc_gr.compile()
        self.abc_ph.compile()
        self._logger.info(f'Collected grapheme dict of {len(self.abc_gr.vocab)} symbols.')
        self._logger.info(f'Collected phoneme dict of {len(self.abc_ph.vocab)} symbols.')

    def graphemes_to_features(self, text):
        features = np.zeros(shape=(len(text), len(self.abc_gr.vocab)))
        for el_i, el in enumerate(text):
            features[el_i, self.abc_gr.vocab[el]] = 1
        return features

    def batches(self, batch_size, is_train):
        source = self.train_data if is_train else self.test_data
        for i in range(0, len(source), batch_size):
            batch_data = source[i:i+batch_size]
            if len(batch_data) == batch_size:
                inputs = []
                texts = []
                for word, pron in batch_data:
                    inputs.append([self.abc_gr.vocab[c] for c in word])
                    texts.append([self.abc_ph.GO] + [self.abc_ph.vocab[p] for p in pron] + [self.abc_ph.EOS])
                inputs_lens = [len(x) for x in inputs]
                texts_lens = [len(x) for x in texts]
                features = np.zeros(shape=(batch_size, max(inputs_lens), len(self.abc_gr.vocab)))
                targets = np.zeros(shape=(batch_size, max(texts_lens)))
                for bi, data in enumerate(zip(inputs, texts)):
                    for el_i, el in enumerate(data[0]):
                        features[bi, el_i, el] = 1
                    targets[bi, :texts_lens[bi]] = data[1]
                yield features, inputs_lens, targets, texts_lens
