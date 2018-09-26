import tensorflow as tf
import matplotlib.pyplot as plt
from librosa import load

from config import config
from units import ABC
from model import E2EModel
from features import mfcc
from timit_source import TimitSource

if __name__ == '__main__':
    data_source = TimitSource('ti-chars', config, ABC(), mfcc)
    data_source.load()
    config.features_num = 20
    with tf.Session() as sess:
        model = E2EModel(config, data_source.abc, sess)
        sess.run(tf.global_variables_initializer())
        model.load()
        path = '/home/akademi4eg/corpuses/sessions/correct/masters/master_sound0.wav'
        while len(path):
            waveform, _ = load(path)
            trans, _, align = model.transcribe(data_source.normalize(mfcc(waveform)))
            trans = trans[:-1]
            print(' '.join(trans))
            if align.ndim > 1:
                plt.imshow(align)
                plt.ylabel(' '.join(trans))
                plt.show()
            text = input('Enter path: ')
