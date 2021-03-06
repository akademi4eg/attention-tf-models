import librosa
import os
import logging
import pickle
import random
import numpy as np
from tqdm import tqdm
from collections import Counter


class TimitSource:
    def __init__(self, name, config, abc, sound_to_features):
        self.sample_rate = 16000
        self.sound_to_features = sound_to_features
        self.name = name
        self.abc = abc
        self._reader_path = config.timit
        self._logger = logging.getLogger('timit')
        try:
            self.load()
        except FileNotFoundError:
            self._train = self._read_utterances('TRAIN')
            self._test = self._read_utterances('TEST')
            self.compile_abc()
            self.save()
        self._logger.info(f'Read {len(self._train)} files for "TRAIN" subset.')
        self._logger.info(f'Read {len(self._test)} files for "TEST" subset.')
        self._cache = {}
        self._mean = None
        self._std = None

    def compile_abc(self):
        for _, c in self._train:
            self.abc.update(c)
        self.abc.compile()
        self._logger.info(f'Collected grapheme dict of {len(self.abc.vocab)} symbols.')

    def save(self):
        with open(f'data/vocab_{self.name}.pickle', 'wb') as f:
            pickle.dump(self.abc, f)
        with open(f'data/vocab_{self.name}_data.pickle', 'wb') as f:
            pickle.dump(self._train, f)
            pickle.dump(self._test, f)

    def load(self):
        with open(f'data/vocab_{self.name}.pickle', 'rb') as f:
            self.abc = pickle.load(f)
        with open(f'data/vocab_{self.name}_data.pickle', 'rb') as f:
            self._train = pickle.load(f)
            self._test = pickle.load(f)
        try:
            with open(f'data/vocab_{self.name}_norm.pickle', 'rb') as f:
                self._mean = pickle.load(f)
                self._std = pickle.load(f)
        except FileNotFoundError:
            self._logger.warning('Failed to load normalization.')

    def _read_utterances(self, subset):
        data = []
        phrases = Counter()
        for root, _, files in tqdm(os.walk(os.path.join(self._reader_path, subset)),
                                   desc=f'Collecting "{subset}"', unit='file'):
            for file in files:
                if not file.endswith('.WAV'):
                    continue
                try:
                    with open(os.path.join(root, file.replace('.WAV', '.TXT')), 'r') as f:
                        text = f.read()
                    text = ' '.join(text.split()[2:]).strip().lower()
                    # remove unnecessary complications
                    for c in {'"', ';', ':', ',', '.', '-', '?', '!'}:
                        text = text.replace(c, '')
                    if phrases[text] > 7:
                        # Prevent bias on two most common phrases.
                        continue
                    phrases[text] += 1
                except FileNotFoundError:
                    continue
                data.append((os.path.join(root, file), text))
        return data

    def batches(self, batch_size, is_train):
        source = self._train if is_train else self._test
        random.shuffle(source)
        for i in range(0, len(source), batch_size):
            batch_data = source[i:i+batch_size]
            if len(batch_data) == batch_size:
                inputs = []
                texts = []
                for path, text in batch_data:
                    if path not in self._cache:
                        waveform, _ = librosa.load(path, self.sample_rate, mono=True, dtype=np.float32)
                        features = self.sound_to_features(waveform)
                        self._cache[path] = features
                    if self._mean is None:
                        self._mean = self._cache[path].mean(axis=0)[np.newaxis, :]
                        self._std = self._cache[path].std(axis=0)[np.newaxis, :]
                        with open(f'data/vocab_{self.name}_norm.pickle', 'wb') as f:
                            pickle.dump(self._mean, f)
                            pickle.dump(self._std, f)
                    inputs.append(self.normalize(self._cache[path]))
                    texts.append([self.abc.GO] + [self.abc.vocab[p] for p in text] + [self.abc.EOS])
                inputs_lens = [x.shape[0] for x in inputs]
                texts_lens = [len(x) for x in texts]
                features_num = inputs[0].shape[1]
                features = np.zeros(shape=(batch_size, max(inputs_lens), features_num))
                targets = np.zeros(shape=(batch_size, max(texts_lens)))
                for bi, data in enumerate(texts):
                    features[bi, :inputs_lens[bi], :] = inputs[bi]
                    targets[bi, :texts_lens[bi]] = data
                yield features, inputs_lens, targets, texts_lens

    def normalize(self, features):
        return (features - self._mean) / self._std
