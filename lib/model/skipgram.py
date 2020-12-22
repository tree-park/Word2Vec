"""
Skip-gram class 생성
with negative sampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGram(nn.Module):
    def __init__(self, batch_size, voc_size, emb_dim):
        super().__init__()
        self.batch_size = batch_size
        self.i_embedding = nn.Embedding(voc_size, emb_dim)
        self.o_embedding = nn.Embedding(voc_size, emb_dim)
        self.pred = torch.tensor([])

    def forward(self, i, o, neg):
        inp_emb = self.i_embedding(i)
        out_emb = self.o_embedding(o)

        self.pred = inp_emb
        pos_val = F.logsigmoid(torch.bmm(out_emb, inp_emb.unsqueeze(2)).squeeze(2).sum(1))
        neg_emb = self.o_embedding(neg)
        neg_val = - F.logsigmoid(torch.bmm(neg_emb, inp_emb.unsqueeze(2)).squeeze(2).sum(1))
        return - (pos_val + neg_val).mean()

    def predict(self, i):
        return self.i_embedding(i)
