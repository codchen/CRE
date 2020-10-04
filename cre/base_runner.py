from collections import defaultdict
from random import choice, sample, shuffle
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import torch
import numpy as np
from sklearn.metrics import precision_recall_curve

from data.alt_dataset import DataSet as AltDataSet
from data.dataset import DataSet

CUDA = torch.cuda.is_available()

def get_pretrained_skipgram(as_dict=False):
    word2vec = {}
    with open('../data/skipgram.txt', 'r') as file:
        for line in file:
            word, rep = line[:-1].split(' ', 1)
            split_rep = rep.split(' ')
            assert not split_rep[-1].isnumeric()
            rep = list(map(float, split_rep[:-1]))
            word2vec[word] = rep
    if as_dict:
        return word2vec
    else:
        word2vec = sorted(list(word2vec.items()))
        vocab = [t[0] for t in word2vec]
        pretrained = [t[1] for t in word2vec]
        return vocab, pretrained

class BaseRunner:
    def __init__(self, use_alt=-1):
        vocab, self.pretrained = get_pretrained_skipgram()
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.dataset = DataSet() if use_alt < 0 else AltDataSet()
        self.rels = sorted(self.dataset.all_relations)
        self.ents = sorted(self.dataset.all_entities)
        self.rel2idx = {rel: idx for idx, rel in enumerate(self.rels)}
        self.idx2rel = {idx: rel for rel, idx in self.rel2idx.items()}
        self.ent2idx = {ent: idx for idx, ent in enumerate(self.ents)}
        self.train_data, self.test_data = self.dataset.split() if use_alt < 0 else self.dataset.load_from_file(use_alt)
        self.enrich_data()
        self.num_examples_per_batch = 5

        self.model = None
        self.optimizer = None
        self.loss = None

    def enrich_data(self):
        pass

    def train(self, epochs=10, batch_size=100, persist_path='../trained_models/na'):
        self.model.train()
        print("start training")

        num = 0
        batch = defaultdict(list)
        for i in range(epochs):
            sample_space = self.sample(self.train_data)
            loss_so_far = 0.0
            for pair, info in sample_space:
                num += 1
                self.add_to_batch(pair, info, batch)
                if num == batch_size:
                    output = self.forward_with_batch(batch)
                    loss = self.loss(output, torch.FloatTensor(batch['target']).cuda() if CUDA else torch.FloatTensor(batch['target']))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    loss_so_far += loss.item()

                    num = 0
                    batch = defaultdict(list)
            print("Epoch {} with loss {}".format(i, loss_so_far))

            predict_scores, facts, predict_rels, pairs = self.predict('tmp')
            precision, recall, _ = precision_recall_curve(facts, predict_scores, pos_label=1)
            self.print_curve(precision, recall)
            self.model.train()

            torch.save(self.model.state_dict(), persist_path)

    def predict(self, result_filename, topk=1):
        torch.no_grad()
        self.model.eval()
        print("start predicting")

        predict_scores = []
        predict_rels = []
        facts = []
        pairs = []
        count = 0
        sample_space = self.sample(self.test_data, True)
        for pair, info in sample_space:
            if len(info.examples) > 500:
                continue
            with torch.no_grad():
                output = self.predict_single(pair, info)
                for _ in range(topk):
                    pairs.append(pair)
                top, top_indices = torch.topk(output.squeeze(), k=topk, dim=-1)
                predict_rels.extend([self.idx2rel[idx] for idx in top_indices.tolist()])
                for idx in top_indices.tolist():
                    if self.idx2rel[idx] in info.relations:
                        facts.append(1)
                    else:
                        facts.append(0)
                predict_scores.extend(top.tolist())

                if CUDA:
                    torch.cuda.empty_cache()
                count += 1
                if count % 1000 == 0:
                    print("Predicted {}".format(count))
        assert len(predict_scores) == len(facts) == len(predict_rels) == len(pairs)
        with open(result_filename, 'w') as file:
            for predict_score, fact, predict_rel, pair in zip(predict_scores, facts, predict_rels, pairs):
                file.write("{},{},{},{},{}\n".format(predict_score, fact, predict_rel, pair[0], pair[1]))
        return predict_scores, facts, predict_rels, pairs

    def predict_single(self, pair, info):
        raise NotImplementedError

    def forward_with_batch(self, batch):
        raise NotImplementedError

    def add_to_batch(self, pair, info, batch):
        raise NotImplementedError

    def sample(self, data, use_all=False):
        sample_space = [(pair, info) for pair, info in data.items() if '[NA]' not in info.relations]
        if use_all:
            sample_space += [(pair, info) for pair, info in data.items() if '[NA]' in info.relations]
        else:
            sample_space += sample([(pair, info) for pair, info in data.items() if '[NA]' in info.relations], len(sample_space) // len(self.rels))
        shuffle(sample_space)
        return sample_space

    def print_curve(self, precision, recall):
        def find_closest_idx(arr, tgt):
            idx = -1
            delta = 100000
            for i, n in enumerate(arr):
                new_delta = abs(n - tgt)
                if new_delta < delta:
                    delta = new_delta
                    idx = i
            return idx
        thresholds = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        results = [precision[find_closest_idx(recall, t)] for t in thresholds]
        print(results)      
