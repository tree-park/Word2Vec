"""
Word2Vec class
    - 학습
    - 평가
"""
import os.path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from lib.data_preprocess import Vocab, TrainSet
from lib.model.cbow import CBOW
from lib.model.skipgram import SkipGram


class Word2Vec:
    def __init__(self, dconf, mconf):
        self.dconf = dconf
        self.mconf = mconf

        self.vocab = Vocab(self.dconf.min_cnt)
        self.voc_size = 0
        self.dataset = None
        self._dataload = None

        self.w2v = None
        self.loss = None
        self.optim = None
        self.lrscheder = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def dataset_form(self, corpus, word2idx):
        raise

    def train(self):
        raise

    def save(self, fname):
        """ save model """
        torch.save({
            'model': self.w2v.state_dict(),
            'optim': self.optim.state_dict(),
            'vocab': self.vocab,
        }, os.path.dirname(__file__) + '/../results/model/' + fname)

    @staticmethod
    def load(fname: str):
        """ load pytorch model """
        checkpoint = torch.load(os.path.dirname(__file__) + '/../results/model/' + fname)
        return checkpoint

    def similarity(self, item1, item2):
        """ return two w2v output's similarity """
        idx1 = self.vocab[item1]
        idx2 = self.vocab[item2]
        emb1 = self.w2v.predict(torch.tensor(idx1).to(self.device)).unsqueeze(0)
        emb2 = self.w2v.predict(torch.tensor(idx2).to(self.device)).unsqueeze(0)
        cos = torch.cosine_similarity(emb1, emb2, dim=1)
        sim = cos
        return sim

    def nearest(self, item, top=10):
        if not self.vocab.idx2word:
            self.vocab.to_idx2word()

        idx = self.vocab[item]
        emb = self.w2v.predict(torch.tensor(idx).to(self.device)).unsqueeze(0)
        sims = torch.mm(emb, self.w2v.i_embedding.weight.transpose(0, 1)).squeeze(0)
        sims = (-sims).sort()[1][0: top + 1]
        tops = []
        for k in range(top):
            tops.append(self.vocab.idx2word[sims[k].item()])
        return tops

    def __len__(self):
        """ return word2vec vector size """
        return len(self.vocab)

    def __getitem__(self, item):
        """ return target item's vector output """
        idx = self.vocab[item]
        return self.w2v.pred(idx)


@torch.no_grad()
def accuracy(pred, target):
    batch_size = pred.size(0)
    pred_idx = torch.max(pred, dim=1)[1]
    acc = 0.00001
    acc += (pred_idx == target).sum().item() / batch_size
    return acc


class CbowModel(Word2Vec):
    def __init__(self, dconf, mconf, mode):
        super(CbowModel, self).__init__(dconf, mconf)
        if mode == 'train':
            self.dataset = TrainSet(self.dconf.data_path, self.vocab, self.dataset_form)
            self.voc_size = len(self.vocab)
            self.w2v = CBOW(self.voc_size, self.dconf.cont_size, self.mconf.emb_dim,
                            self.mconf.hid_dim)
        else:
            checkpoint = self.load(dconf.saved_file)
            if self.optim and mode is 'retrain':
                self.optim.load_state_dict(checkpoint['optim'])
            self.vocab = checkpoint['vocab']
            self.vocab.to_idx2word()
            self.voc_size = len(self.vocab)
            self.w2v = CBOW(self.voc_size, self.dconf.cont_size, self.mconf.emb_dim,
                            self.mconf.hid_dim)
        self.w2v.to(self.device)

    def dataset_form(self, corpus, word2idx):
        rst = []
        for sent in corpus:
            sent = [word2idx[x] for x in sent if word2idx[x]]
            for i in range(1, len(sent) - 1):
                wrd = sent[i]
                trg_f = sent[i - 1]
                trg_b = sent[i + 1]
                rst.append(([trg_f, trg_b], wrd))
        return rst

    def train(self):
        """ epoch만큼 학습 """
        assert self.dataset
        self._dataload = DataLoader(self.dataset,
                                    batch_size=self.mconf.batch_size,
                                    num_workers=0)

        self.loss = nn.NLLLoss().to(self.device)
        self.optim = optim.Adam(params=self.w2v.parameters(), lr=self.mconf.lr)
        self.lrscheder = optim.lr_scheduler.ReduceLROnPlateau(self.optim, patience=5)

        for epoch in tqdm(range(self.mconf.epoch), desc='epoch'):
            total_loss = 0
            total_acc = 0
            self.w2v.train()
            for i, batch in enumerate(self._dataload):
                word, target = map(lambda x: x.to(self.device), batch)
                self.optim.zero_grad()
                pred = self.w2v(word)
                b_loss = self.loss(pred, target)
                b_loss.backward()
                self.optim.step()
                total_loss += b_loss.item()
            if epoch % 10 == 0:
                self.save('trained.pth')
            print(f'\tEpoch {epoch + 1}\tTrain Loss: {total_loss / len(word):.3f}'
                  f' | Acc: {total_acc / len(word):.3f}')


class SkipGramModel(Word2Vec):
    def __init__(self, dconf, mconf, mode):
        super(SkipGramModel, self).__init__(dconf, mconf)
        if mode == 'train':
            self.dataset = TrainSet(self.dconf.data_path, self.vocab, self.dataset_form)
            self.voc_size = len(self.vocab)
            self.w2v = SkipGram(self.voc_size, self.mconf.emb_dim)
        else:
            checkpoint = self.load(dconf.saved_file)
            if self.optim and mode is 'retrain':
                self.optim.load_state_dict(checkpoint['optim'])
            self.vocab = checkpoint['vocab']
            self.vocab.to_idx2word()
            self.voc_size = len(self.vocab)
            self.w2v = SkipGram(self.voc_size, self.mconf.emb_dim)
            self.w2v.load_state_dict(checkpoint['model'])
        self.w2v.to(self.device)

    def dataset_form(self, corpus, word2idx):
        """ return input word, target words, negative samples """
        pos_samples = get_sub_sample(corpus, word2idx)
        neg_samples = get_neg_sample(word2idx, len(pos_samples))
        rst = [(i[0][0], i[0][1], i[1]) for i in zip(pos_samples, neg_samples)]
        return rst

    def train(self):
        """ Skip gram """
        self._dataload = DataLoader(self.dataset,
                                    batch_size=self.mconf.batch_size,
                                    num_workers=0)

        self.loss = nn.NLLLoss().to(self.device)
        self.optim = optim.Adam(params=self.w2v.parameters(), lr=self.mconf.lr)
        self.lrscheder = optim.lr_scheduler.ReduceLROnPlateau(self.optim, patience=5)

        for epoch in tqdm(range(self.mconf.epoch), desc='epoch'):
            total_loss = 0
            self.w2v.train()
            for i, batch in enumerate(self._dataload):
                inp, out, neg = map(lambda x: x.to(self.device), batch)
                self.optim.zero_grad()
                loss = self.w2v(inp, out, neg)
                loss.backward()
                self.optim.step()
                total_loss += loss.item()
            if epoch % 10 == 0:
                self.save('trained.pth')
            print(f'\tEpoch {epoch + 1}\tTrain Loss: {total_loss / len(inp):.3f}')


def get_sub_sample(corpus, word2idx):
    pos_samples = []
    for sent in corpus:
        sent = [word2idx[x] for x in sent if word2idx[x]]
        for i in range(1, len(sent) - 1):
            i_w = sent[i]
            o_w1 = sent[i - 1]
            o_w2 = sent[i + 1]
            pos_samples.append((i_w, [o_w1, o_w2]))
    return pos_samples


def get_neg_sample(word2idx, total_size, t=10 ** -5, k=20):
    vfreq = list(word2idx.freq.values())
    tmp = [x**(3/4) for x in vfreq]
    z_i = [x / sum(tmp) for x in tmp]
    # noise_dstr = [1 - np.sqrt((z / t)) for z in z_i]
    # noise_dstr = [(1 + np.sqrt(z / t)) * (t / z) for z in z_i]
    neg_samples = []
    for _ in range(total_size):
        ne_samps = np.random.choice(list(word2idx.freq.keys()), k, p=z_i)
        neg_samples.append(ne_samps)
    return neg_samples
