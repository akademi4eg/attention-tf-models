import numpy as np
import librosa


SAMPLE_RATE = 16000


def raw_waveform(waveform):
    return waveform[:, np.newaxis]


def mfcc(waveform):
    mfcc = librosa.feature.mfcc(y=waveform, sr=SAMPLE_RATE, n_mfcc=20, n_mels=64,
                                n_fft=256, hop_length=128)
    return mfcc.transpose()


def log_melspectrogram(waveform):
    spec = librosa.feature.melspectrogram(y=waveform, sr=SAMPLE_RATE, n_mels=64,
                                          n_fft=256, hop_length=128)
    return np.log(spec).transpose()
