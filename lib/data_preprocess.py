"""
- create custom Dataset class
- Vocab - class로 따로관리
- Tokenization
- Padding
"""
import re
import torch
from torch.utils.data import Dataset
import nltk
from nltk.stem import WordNetLemmatizer

from .data_handle import load_data

nltk.download('wordnet')


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
        # word, content = self._data[idx]
        # return torch.tensor([word[0], word[1]]), torch.tensor(content)
        return [i for i in map(lambda x: torch.tensor(x), [x for x in self._data[idx]])]


class Vocab:
    UNKNOWN = '[UKN]'
    WORD2IDX = {UNKNOWN: 0}

    def __init__(self, min_cnt):
        self.min_cnt = min_cnt
        self.excepts = '?!.#$%^&*'
        self.word2idx = self.WORD2IDX
        self.idx2word = {}
        self.freq = {}
        self.lemmatizer = WordNetLemmatizer()

    def load(self, corpus: list):
        vocabs = {}
        for sent in corpus:
            for word in sent:
                if word not in vocabs.keys():
                    vocabs[word] = 0
                vocabs[word] += 1
        self.word2idx.update({w: idx for idx, w in enumerate(vocabs.keys(), start=len(self.word2idx))})
        self.freq = {self.word2idx[k]: v for k, v in vocabs.items()}
        return self

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
        word = self.lemmatizer.lemmatize(item.lower())
        try:
            return self.word2idx[word]
        except KeyError:
            return self.word2idx[self.UNKNOWN]


def _preprocessor(corpus: list):
    result = []
    lemmatizer = WordNetLemmatizer()
    for line in corpus:
        sents = _to_sentence(line)
        for s in sents:
            words = _to_word(s, lemmatizer)
            result.append(words)
    return result


SENT_END = ".?!"
SENT_SEP = r'(%s)+(\s|\n)' % SENT_END


def _to_sentence(a_line: str) -> list:
    rst = re.split(SENT_SEP, a_line)
    return rst


def _to_word(a_sent: str, lemmatizer: callable(str)) -> list:
    rst = []
    for word in a_sent.split(' '):
        if word.strip() in ',./?!':
            continue
        word = word.lower()
        for rm in ['!', '?', '.', '\'', '\"', '*']:
            word = word.replace(rm, '')
        if word in ['a', 'an', 'i', 'she', 'he', 'the', 'they', 'of', 'so', 'to']:
            continue
        word = lemmatizer.lemmatize(word)
        rst.append(word)
    return rst
