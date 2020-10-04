from math import sqrt

import torch
from torch import nn

CUDA = torch.cuda.is_available()

class M2R(nn.Module):
    def __init__(self, n_v, n_r, k=50):
        super().__init__()
        self.W = nn.Parameter(torch.FloatTensor(k, n_v).normal_(mean=0.0, std=1.0/k))
        self.r = nn.Parameter(torch.FloatTensor(k, n_r).normal_(mean=0.0, std=1.0/k))
        self.relu = nn.ReLU()

    def forward(self, Mi, Mj, ri, rp):
        loss = torch.matmul(
            torch.matmul(self.W, Mj.t()).t(),
            torch.index_select(self.r, 1, rp),
        ).diag()
        loss = loss - torch.matmul(
            torch.matmul(self.W, Mi.t()).t(),
            torch.index_select(self.r, 1, ri),
        ).diag()
        loss = loss + 1.0
        return self.relu(loss).mean()

    def score(self, M, topk=1):
        # M: N * n_v
        t1 = torch.matmul(self.W, M.t()).t() # N * k
        t2 = torch.matmul(t1, self.r) # N * n_r
        t3 = torch.sum(t2, dim=0) # n_r
        t4 = torch.topk(t3, k=topk, dim=-1)
        return t4

    def normalize(self):
        self.W = self.normalize_param(self.W)
        self.r = self.normalize_param(self.r)

    def normalize_param(self, param):
        l2_norm = torch.max(torch.tensor(1.0).cuda() if CUDA else torch.tensor(1.0), torch.norm(param, p=2, dim=0))
        return nn.Parameter(
            param / l2_norm.unsqueeze(0),
        )

class TransE(nn.Module):
    def __init__(self, n_e, n_r, k=50, gamma=40, l2_penalty=0.05):
        super().__init__()
        self.E = nn.Parameter(
            torch.FloatTensor(n_e, k).uniform_(-6/sqrt(k), 6/sqrt(k)).cuda() if CUDA else torch.FloatTensor(n_e, k).uniform_(-6/sqrt(k), 6/sqrt(k))
        )
        self.R = nn.Parameter(
            torch.FloatTensor(n_r, k).uniform_(-6/sqrt(k), 6/sqrt(k)).cuda() if CUDA else torch.FloatTensor(n_r, k).uniform_(-6/sqrt(k), 6/sqrt(k))
        )
        self.relu = nn.ReLU()

        self.n_r = n_r
        self.gamma = gamma
        self.l2_penalty = l2_penalty

    def normalize(self):
        l2_norm = torch.norm(self.E, p=2, dim=1)
        self.E = nn.Parameter(self.E.clone() / torch.max(l2_norm, other=torch.tensor([1.0]).cuda() if CUDA else torch.tensor([1.0])).unsqueeze(1))

    def distance(self, h, l, t):
        hs = torch.index_select(self.E, dim=0, index=h)
        ls = torch.index_select(self.R, dim=0, index=l)
        ts = torch.index_select(self.E, dim=0, index=t)

        diff = hs + ls - ts
        return torch.norm(diff, p=2, dim=1)

    def forward(self, h, t, hp, tp, l):
        result_a = self.distance(h, l, t)
        result_b = result_a - self.distance(hp, l, tp)
        result_c = result_b + 1.0
        return torch.mean(self.relu(result_c)) # + self.l2_penalty * (torch.norm(self.E, p=2) + torch.norm(self.R, p=2))

    def score(self, h, t, l):
        s = self.distance(
            torch.LongTensor([h]).cuda() if CUDA else torch.LongTensor([h]),
            torch.LongTensor([l]).cuda() if CUDA else torch.LongTensor([l]),
            torch.LongTensor([t]).cuda() if CUDA else torch.LongTensor([t])
        ).squeeze()
        all_distances = self.distance(
            torch.LongTensor([h] * self.n_r).cuda() if CUDA else torch.LongTensor([h] * self.n_r),
            torch.LongTensor(list(range(self.n_r))).cuda() if CUDA else torch.LongTensor(list(range(self.n_r))),
            torch.LongTensor([t] * self.n_r).cuda() if CUDA else torch.LongTensor([t] * self.n_r),
        )
        num_r_closer = torch.sum(all_distances < s).item()
        if num_r_closer >= self.gamma:
            return 0.0
        else:
            return 1.0
