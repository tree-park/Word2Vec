"""
CBOW 모델 선언
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CBOW(nn.Module):
    def __init__(self, voc_size, cont_size, emb_dim, hid_dim):
        super().__init__()
        self.embedding = nn.Embedding(voc_size, emb_dim)
        self.inp_layer = nn.Linear(cont_size * 2 * emb_dim, hid_dim)
        self.out_layer = nn.Linear(hid_dim, voc_size)

    def forward(self, inputs):
        embed = self.embedding(inputs).view((inputs.size(0), -1))
        inp = F.relu(self.inp_layer(embed))
        out = self.out_layer(inp)
        return F.log_softmax(out, dim=1)

    def predict(self, idx):
        return self.embedding(idx)
