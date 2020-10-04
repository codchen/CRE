from ast import literal_eval
from random import randint, sample
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import numpy as np
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

from weston.model import M2R, TransE
from data.alt_dataset import DataSet as AltDataSet
from data.dataset import DataSet

CUDA = torch.cuda.is_available()

class WestonDataSet(DataSet):
    def __init__(self):
        super().__init__()
        self.sentences_to_dep_path = {}
        with open('../data/dep_paths.csv', 'r') as file:
            for line in file:
                sentence, dep_path = line[:-1].split('|')
                dep_path = literal_eval(dep_path)
                self.sentences_to_dep_path[sentence] = dep_path
        self.sentences_to_pos_tags = {}
        with open('../data/pos_tags.csv', 'r') as file:
            for line in file:
                sentence, pos_tags = line[:-1].split('|')
                pos_tags = literal_eval(pos_tags)
                self.sentences_to_pos_tags[sentence] = pos_tags

    def featurize(self, ex):
        sentence = ex.pre + ' SPETT1 ' + ex.mid + ' SPETT2 ' + ex.post
        def get_dep_path():
            dep_path = self.sentences_to_dep_path[sentence]
            if ex.h_idx < ex.t_idx:
                return dep_path
            return [d[2:] if d.startswith('d-') else 'd-' + d for d in dep_path]
        return get_dep_path() + self.sentences_to_pos_tags[sentence] + [ex.head, ex.tail]

class WestonAltDataSet(AltDataSet):
    def __init__(self):
        super().__init__()
        self.sentences_to_dep_path = {}
        with open('../data/dep_paths.csv', 'r') as file:
            for line in file:
                sentence, dep_path = line[:-1].split('|')
                dep_path = literal_eval(dep_path)
                self.sentences_to_dep_path[sentence] = dep_path
        self.sentences_to_pos_tags = {}
        with open('../data/pos_tags.csv', 'r') as file:
            for line in file:
                sentence, pos_tags = line[:-1].split('|')
                pos_tags = literal_eval(pos_tags)
                self.sentences_to_pos_tags[sentence] = pos_tags

    def featurize(self, ex):
        sentence = ex.pre + ' SPETT1 ' + ex.mid + ' SPETT2 ' + ex.post
        def get_dep_path():
            if sentence not in self.sentences_to_dep_path:
                return []
            dep_path = self.sentences_to_dep_path[sentence]
            if ex.h_idx < ex.t_idx:
                return dep_path
            return [d[2:] if d.startswith('d-') else 'd-' + d for d in dep_path]
        return get_dep_path() + self.sentences_to_pos_tags.get(sentence, []) + [ex.head, ex.tail]

class Runner:
    def __init__(self, k=50, k_transe=50, gamma=40, m2r_rate=0.001, kb_rate=0.1, use_alt=-1, load_m2r_from=None, load_transe_from=None):
        self.dataset = WestonDataSet() if use_alt < 0 else WestonAltDataSet()
        self.dataset.featurize_all()
        self.train_data, self.test_data = self.dataset.split() if use_alt < 0 else self.dataset.load_from_file(use_alt)
        self.vocab = sorted(self.dataset.vocab)
        self.rels = sorted(self.dataset.all_relations)
        self.ents = sorted(self.dataset.all_entities)
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.rel2idx = {rel: idx for idx, rel in enumerate(self.rels)}
        self.idx2rel = {idx: rel for rel, idx in self.rel2idx.items()}
        self.ent2idx = {ent: idx for idx, ent in enumerate(self.ents)}

        self.m2r = M2R(n_v=len(self.vocab), n_r=len(self.rels), k=k)
        if CUDA:
            self.m2r = self.m2r.cuda()
        if load_m2r_from:
            self.m2r.load_state_dict(torch.load(load_m2r_from))
        self.m2r_rate = m2r_rate
        self.transe = TransE(n_e=len(self.ents), n_r=len(self.rels), k=k_transe, gamma=gamma)
        if CUDA:
            self.transe = self.transe.cuda()
        if load_transe_from:
            self.transe.load_state_dict(torch.load(load_transe_from))
        self.kb_rate = kb_rate

    def features_to_bin(self, features):
        result = np.zeros(len(self.vocab))
        for feature in features:
            if feature in self.word2idx:
                result[self.word2idx[feature]] = 1
        return result
        
    def train_m2r(self, iters=10, persist_path='../trained_models/weston_m2r.mod'):
        self.m2r.train()

        for i in range(iters):
            # necessary because weights are swapped out every iter
            self.m2r_optimizer = optim.Adam(
                self.m2r.parameters(),
                lr=self.m2r_rate,
            )

            mis = []
            mjs = []
            ris = []
            rps = []

            sample_space = [info for info in self.train_data.values() if '[NA]' not in info.relations]
            sample_space += sample([info for info in self.train_data.values() if '[NA]' in info.relations], len(sample_space) // len(self.rels))
            for _ in range(1000):
                exi, exj = sample(sample_space, 2)
                mi = sample(exi.examples, 1)[0]
                mj = sample(exj.examples, 1)[0]
                ri = sample(exi.relations, 1)[0]
                remaining_relations = set(self.rels) - set([ri] + exj.relations)
                rp = sample(remaining_relations, 1)[0]
                mis.append(self.features_to_bin(mi.feature))
                mjs.append(self.features_to_bin(mj.feature))
                ris.append(self.rel2idx[ri])
                rps.append(self.rel2idx[rp])

            loss = self.m2r(
                torch.from_numpy(np.asarray(mis)).float().cuda() if CUDA else torch.from_numpy(np.asarray(mis)).float(),
                torch.from_numpy(np.asarray(mjs)).float().cuda() if CUDA else torch.from_numpy(np.asarray(mjs)).float(),
                torch.from_numpy(np.asarray(ris)).cuda() if CUDA else torch.from_numpy(np.asarray(ris)),
                torch.from_numpy(np.asarray(rps)).cuda() if CUDA else torch.from_numpy(np.asarray(rps)),
            )
            self.m2r_optimizer.zero_grad()
            loss.backward()
            self.m2r_optimizer.step()
            self.m2r.normalize()

            if (i + 1) % 100 == 0:
                print("M2R Iter {} with loss {}".format(i, loss.item()))
            torch.save(self.m2r.state_dict(), persist_path)

    def train_transe(self, iters=10, persist_path='../trained_models/weston_transe.mod'):
        self.transe.train()

        training_triplets = []
        for pair, info in self.train_data.items():
            for relation in info.relations:
                if relation == '[NA]':
                    continue
                training_triplets.append((self.ent2idx[pair[0]], self.ent2idx[pair[1]], self.rel2idx[relation]))

        for i in range(iters):
            # necessary because weights are swapped out every iter
            self.transe_optimizer = optim.Adam(
                self.transe.parameters(),
                lr=self.kb_rate,
            )

            triplet_sample = sample(training_triplets, 100)

            hs, ts, ls, hps, tps = [], [], [], [], []
            for triplet in triplet_sample:
                h, t, l = triplet
                if randint(0, 1) == 0:
                    hp_candidates = set(list(self.ent2idx.values())) - set([h])
                    hp = sample(hp_candidates, 1)[0]
                    tp = t
                else:
                    tp_candidates = set(list(self.ent2idx.values())) - set([t])
                    tp = sample(tp_candidates, 1)[0]
                    hp = h
                hs.append(h)
                ts.append(t)
                ls.append(l)
                hps.append(hp)
                tps.append(tp)
            loss = self.transe(
                torch.LongTensor(hs).cuda() if CUDA else torch.LongTensor(hs),
                torch.LongTensor(ts).cuda() if CUDA else torch.LongTensor(ts),
                torch.LongTensor(hps).cuda() if CUDA else torch.LongTensor(hps),
                torch.LongTensor(tps).cuda() if CUDA else torch.LongTensor(tps),
                torch.LongTensor(ls).cuda() if CUDA else torch.LongTensor(ls),
            )

            self.transe_optimizer.zero_grad()
            loss.backward()
            self.transe_optimizer.step()

            self.transe.normalize()

            if (i + 1) % 100 == 0:
                print("TransE Iter {} with loss {}".format(i, loss.item() / 1000))
            torch.save(self.transe.state_dict(), persist_path)

    def predict(self, result_filename, topk=1):
        torch.no_grad()
        self.m2r.eval()
        self.transe.eval()

        predict_scores = []
        predict_rels = []
        facts = []
        pairs = []
        count = 0
        to_be_tested = [t for t in self.test_data.items() if '[NA]' not in t[1].relations]
        to_be_tested += [t for t in self.test_data.items() if '[NA]' in t[1].relations]
        for pair, info in to_be_tested:
            bins = np.asarray([self.features_to_bin(m.feature) for m in info.examples])

            for _ in range(topk):
                pairs.append(pair)
            top, top_indices = self.m2r.score(torch.from_numpy(bins).float().cuda() if CUDA else torch.from_numpy(bins).float(), topk=topk)
            predict_rels.extend([self.idx2rel[idx] for idx in top_indices.tolist()])
            rel_indices = top_indices.tolist()
            for idx in rel_indices:
                if self.idx2rel[idx] in info.relations:
                    facts.append(1)
                else:
                    facts.append(0)
            scores = top.tolist()
            if rel_indices[0] != self.rel2idx['[NA]']:
                scores[0] += self.transe.score(self.ent2idx[pair[0]], self.ent2idx[pair[1]], rel_indices[0])
            predict_scores.extend(scores)

            count += 1
            if count % 1000 == 0:
                print("Predict Iter {}".format(count))
        assert len(predict_scores) == len(facts) == len(predict_rels) == len(pairs)
        with open(result_filename, 'w') as file:
            for predict_score, fact, predict_rel, pair in zip(predict_scores, facts, predict_rels, pairs):
                file.write("{},{},{},{},{}\n".format(predict_score, fact, predict_rel, pair[0], pair[1]))
        return predict_scores, facts, predict_rels, pairs
