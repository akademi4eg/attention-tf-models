from collections import Counter


class ABC:
    def __init__(self):
        self.GO = 1
        self.EOS = 0
        self._units = set()
        self.vocab = {}
        self.rev_vocab = {}

    def update(self, string):
        for c in string:
            self._units.add(c)

    def get_char(self, index):
        if index in self.rev_vocab:
            return self.rev_vocab[index]
        return '<UNK>'

    def compile(self):
        self.vocab = {c: ic for ic, c in enumerate(self._units, start=2)}
        self.vocab['<GO>'] = self.GO
        self.vocab['<EOS>'] = self.EOS
        self.rev_vocab = {ic: c for c, ic in self.vocab.items()}


class BPE(ABC):
    def __init__(self, vocab_size):
        super().__init__()
        self._all_data = []
        self.vocab_size = vocab_size

    def update(self, string):
        super().update(string)
        self._all_data.append(string)

    def compile(self):
        super().compile()
        for _ in range(self.vocab_size):
            pass
