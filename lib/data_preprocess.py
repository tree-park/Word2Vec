"""
- create custom Dataset class
- Vocab - class로 따로관리
- Tokenization
- Padding
"""
import re
import torch
from torch.utils.data import Dataset
from .data_handle import load_data


class TrainSet(Dataset):

    def __init__(self, filepath: str, vocab, form_func):
        self._vocab = vocab

        self._corpus = _preprocessor(load_data(filepath))
        self.word2idx = self._vocab.load(self._corpus)
        self._data = form_func(self._corpus, self.word2idx)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx: int):
        """ return input word and target word """
        word, content = self._data[idx]
        return torch.tensor([word[0], word[1]]), torch.tensor(content)


class Vocab:
    UNKNOWN = '[UKN]'
    WORD2IDX = {UNKNOWN: 0}

    def __init__(self, min_cnt):
        self.min_cnt = min_cnt
        self.excepts = '?!.#$%^&*'
        self.word2idx = self.WORD2IDX
        self.idx2word = {}
        self.freq = {}

    def load(self, corpus: list):
        vocabs = {}
        for sent in corpus:
            for word in sent:
                if word not in vocabs.keys():
                    vocabs[word] = 0
                vocabs[word] += 1

        vocabs = vocabs.keys()
        self.freq = vocabs
        self.word2idx = {w: idx for idx, w in enumerate(vocabs)}
        return self.word2idx

    def _vocabs_filter(self, v, cnt):
        if cnt < self.min_cnt:
            return
        if v in self.excepts:
            return
        return v

    def to_idx2word(self):
        self.idx2word = {idx: w for w, idx in self.word2idx.items()}

    def get_word(self, idx):
        return self.idx2word[idx]

    def __len__(self):
        return len(self.word2idx)

    def __getitem__(self, item):
        try:
            return self.word2idx[item]
        except KeyError:
            return self.word2idx[self.UNKNOWN]


def _preprocessor(corpus: list):
    result = []
    for line in corpus:
        sents = _to_sentence(line)
        for s in sents:
            words = _to_word(s)
            result.append(words)
    return result


SENT_END = ".?!"
SENT_SEP = r'(%s)+(\s|\n)' % SENT_END


def _to_sentence(a_line: str) -> list:
    rst = re.split(SENT_SEP, a_line)
    return rst


def _to_word(a_sent: str) -> list:
    rst = []
    for word in a_sent.split(' '):
        if word.strip() in ',./?!':
            continue
        rst.append(word.lower())
        # 불용어 처리 등등...
    return rst
