import tensorflow as tf
import matplotlib.pyplot as plt

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
        text = 'tensorflow'
        while len(text):
            trans, _, align = model.transcribe(data_source.graphemes_to_features(text))
            trans = trans[:-1]
            print(' '.join(trans))
            if align.ndim > 1:
                plt.imshow(align)
                plt.ylabel(' '.join(trans))
                plt.xlabel(text)
                plt.show()
            text = input('Enter text: ')
