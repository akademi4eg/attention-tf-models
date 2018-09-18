# Tensorflow encoder-decoder with attention.
## Model overview
### Preprocessing layers
Optional convolution and pooling layers.
### Encoder
Pyramidal stack of BiLSTM layers. Outputs and states of forward and backward cells would be concatenated.
### Decoder
Stack of LSTMs with Bahdanau attention. Decoder is unidirectional and thus operates on 2 x cell_units.

## Experiments
### Grapheme to phoneme
Simple model to test the pipeline.
Translates english words to phones. Trained on CMUDict. With most of configurations you should expect to get below 50% WER.
Pyramidal stacks don't seem to be optimal architecture choice for this task, so getting below 40% would be challenging.
To do training, run `train_g2p.py`. To test transcription on some words, run `transcribe_g2p.py`.
Example config for G2P:
```python
config.batch_size = 512
config.cell_units = 64
config.encoder_layers = 3
config.decoder_layers = 2
config.attention_units = 128
config.decoder_emb_size = 128
config.pool_step = 1
config.conv_units = 0
config.pass_state = True
config.max_iter = 50
config.learn_rate = 1e-4
config.clip_norm = 20
config.sampling_prob = 0.1
config.dropout_in = 0.0
config.dropout_out = 0.1
```

## Useful links
* [WIndQAQ implementation of LAS model](https://github.com/WindQAQ/listen-attend-and-spell)