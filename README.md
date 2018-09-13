# Tensorflow encoder-decoder with attention.
## Model overview
### Preprocessing layers
Optional convolution and pooling layers.
### Encoder
Stack of BiLSTM layers. Outputs and states of forward and backward cells would be concatenated.
### Decoder
Stack of LSTMs with Bahdanau attention. Decoder is unidirectional and thus operates on 2 x cell_units.

## Experiments
### Grapheme to phoneme
Simple model to test the pipeline.
Translates english words to phones. Trained on CMUDict.
Successful training should get you about 8% character error rate and 34% word error rate.
That's on version that doesn't take into account stress.
To do training, run `train_g2p.py`. To test transcription on some words, run `transcribe_g2p.py`.

## Useful links
* [WIndQAQ implementation of LAS model](https://github.com/WindQAQ/listen-attend-and-spell)