import tensorflow as tf

from config import config
from units import ABC
from model import E2EModel
from cmu_source import CMUDataSource

if __name__ == '__main__':
    data_source = CMUDataSource('chars', ABC(), ABC(), False)
    config.features_num = len(data_source.abc_gr.vocab)
    with tf.Session() as sess:
        model = E2EModel(config, data_source.abc_ph, sess)
        sess.run(tf.global_variables_initializer())
        model.load()
        text = 'hello'
        while len(text):
            trans, _ = model.transcribe(data_source.graphemes_to_features(text))
            print(' '.join(trans))
            text = input('Enter text: ')
