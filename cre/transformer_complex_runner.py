from collections import Counter
from random import choice
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import torch
from torch import nn
from torch import optim

from cre.base_runner import BaseRunner
from cre.transformer_complex import TransformerComplex

CUDA = torch.cuda.is_available()

class TransformerComplexRunner(BaseRunner):
    def __init__(self, use_alt=-1, load_model=None):
        super().__init__(use_alt=use_alt)
        self.model = TransformerComplex(
            num_entities=len(self.ents),
            num_relations=len(self.rels),
            knowledge_dim=100,
            pretrained=self.pretrained,
        )
        if CUDA:
            self.model = self.model.cuda()
        if load_model:
            self.model.load_state_dict(torch.load(load_model))
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=0.0001,
            weight_decay=0.0000005,
        )
        self.loss = nn.BCELoss()
        self.num_examples_per_batch = 10

    def enrich_data(self):
        self.max_len = 0
        for info in list(self.train_data.values()) + list(self.test_data.values()):
            for example in info.examples:
                example.indices = (
                    [self.word2idx[word] if word in self.word2idx else 0 for word in example.pre.split(' ')] +
                    [0] +
                    [self.word2idx[word] if word in self.word2idx else 0 for word in example.mid.split(' ')] +
                    [0] +
                    [self.word2idx[word] if word in self.word2idx else 0 for word in example.post.split(' ')]
                )
                forward = example.h_idx < example.t_idx
                example.h_idx = len(example.pre.split(' '))
                example.t_idx = example.h_idx + len(example.mid.split(' ')) + 1
                if not forward:
                    tmp = example.h_idx
                    example.h_idx = example.t_idx
                    example.t_idx = tmp
                self.max_len = max(self.max_len, len(example.indices))
        for info in list(self.train_data.values()) + list(self.test_data.values()):
            for example in info.examples:
                example.indices += [0] * (self.max_len - len(example.indices))

    def forward_with_batch(self, batch):
        return self.model(
            X=torch.LongTensor(batch['X']).cuda() if CUDA else torch.LongTensor(batch['X']),
            heads=batch['heads'],
            tails=batch['tails'],
            h_idx=batch['h_idx'],
            t_idx=batch['t_idx'],
        )

    def predict_single(self, pair, info):
        return self.model.predict(
            X=torch.LongTensor([[ex.indices for ex in info.examples]]).cuda() if CUDA else torch.LongTensor([[ex.indices for ex in info.examples]]),
            heads=[self.ent2idx[pair[0]]],
            tails=[self.ent2idx[pair[1]]],
            h_idx=[[ex.h_idx for ex in info.examples]],
            t_idx=[[ex.t_idx for ex in info.examples]],
        )

    def add_to_batch(self, pair, info, batch):
        batch['heads'].append(self.ent2idx[pair[0]])
        batch['tails'].append(self.ent2idx[pair[1]])
        examples = [choice(info.examples) for _ in range(self.num_examples_per_batch)]
        batch['X'].append([ex.indices for ex in examples])
        batch['h_idx'].append([ex.h_idx for ex in examples])
        batch['t_idx'].append([ex.t_idx for ex in examples])

        batch['target'].append(
            [0] * len(self.rel2idx)
        )
        for rel in info.relations:
            batch['target'][-1][self.rel2idx[rel]] = 1.0 / len(info.relations)
