from collections import defaultdict
from random import choice, sample, shuffle
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

from han.model import Joint
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

class Runner:
    def __init__(self, nn_lr=0.005, kg_lr=0.0005, l2_penalty=0.0001, conv_window=3, hidden_dim=230, k=50, use_alt=-1, load_model=None):
        self.vocab, pretrained = get_pretrained_skipgram()
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}

        self.dataset = DataSet() if use_alt < 0 else AltDataSet()
        self.dataset.featurize_all()
        self.rels = sorted(self.dataset.all_relations)
        self.ents = sorted(self.dataset.all_entities)
        self.rel2idx = {rel: idx for idx, rel in enumerate(self.rels)}
        self.idx2rel = {idx: rel for rel, idx in self.rel2idx.items()}
        self.ent2idx = {ent: idx for idx, ent in enumerate(self.ents)}
        self.train_data, self.test_data = self.dataset.split() if use_alt < 0 else self.dataset.load_from_file(use_alt)
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
                self.set_position(example)
                self.max_len = max(self.max_len, len(example.indices))
        for info in list(self.train_data.values()) + list(self.test_data.values()):
            for example in info.examples:
                self.pad_example(example)

        self.model = Joint(
            num_relations=len(self.rels),
            num_entities=len(self.ents),
            pretrained=pretrained,
            conv_window=conv_window,
            hidden_dim=hidden_dim,
            k=k,
        )
        if CUDA:
            self.model = self.model.cuda()
        if load_model:
            self.model.load_state_dict(torch.load(load_model))
        self.nn_optimizer = optim.Adam(
            self.model.parameters(),
            lr=nn_lr,
            weight_decay=l2_penalty,
        )
        self.kg_optimizer = optim.Adam(
            self.model.parameters(),
            lr=kg_lr,
            weight_decay=l2_penalty,
        )

    def set_position(self, example):
        pre_len = len(example.pre.split(' '))
        mid_len = len(example.mid.split(' '))
        post_len = len(example.post.split(' '))
        example.pos_pos_1 = [0] * pre_len + list(range(mid_len + post_len + 2))
        example.neg_pos_1 = list(range(pre_len + 1))
        example.neg_pos_1.reverse()
        example.neg_pos_1 += [0] * (mid_len + post_len + 1)
        example.entity_pos_1 = [0] * (pre_len + mid_len + post_len + 2)
        example.entity_pos_1[pre_len] = 1
        example.pos_pos_2 = [0] * (pre_len + mid_len + 1) + list(range(post_len + 1))
        example.neg_pos_2 = list(range(pre_len + mid_len + 2))
        example.neg_pos_2.reverse()
        example.neg_pos_2 += [0] * post_len
        example.entity_pos_2 = [0] * (pre_len + mid_len + post_len + 2)
        example.entity_pos_2[pre_len + mid_len + 1] = 1

        if example.h_idx > example.t_idx:
            tmp = example.pos_pos_1
            example.pos_pos_1 = example.pos_pos_2
            example.pos_pos_2 = tmp
            tmp = example.neg_pos_1
            example.neg_pos_1 = example.neg_pos_2
            example.neg_pos_2 = tmp
            tmp = example.entity_pos_1
            example.entity_pos_1 = example.entity_pos_2
            example.entity_pos_2 = tmp
        self.cap_position(example.pos_pos_1)
        self.cap_position(example.neg_pos_1)
        self.cap_position(example.pos_pos_2)
        self.cap_position(example.neg_pos_2)
        self.cap_position(example.entity_pos_1)
        self.cap_position(example.entity_pos_2)

    def cap_position(self, position):
        for i in range(len(position)):
            if position[i] > 99:
                position[i] = 99

    def pad_example(self, example):
        example.indices += [0] * (self.max_len - len(example.indices))
        example.pos_pos_1 += [0] * (self.max_len - len(example.pos_pos_1))
        example.neg_pos_1 += [0] * (self.max_len - len(example.neg_pos_1))
        example.entity_pos_1 += [0] * (self.max_len - len(example.entity_pos_1))
        example.pos_pos_2 += [0] * (self.max_len - len(example.pos_pos_2))
        example.neg_pos_2 += [0] * (self.max_len - len(example.neg_pos_2))
        example.entity_pos_2 += [0] * (self.max_len - len(example.entity_pos_2))
        assert len(example.pos_pos_1) == self.max_len

    def train(self, iters=10, batch_size=100, persist_path='../trained_models/han.mod'):
        print("start training")
        self.model.train()

        rel2pairs = {}
        for pair, info in self.train_data.items():
            for rel in info.relations:
                if rel in rel2pairs:
                    rel2pairs[rel].append(pair)
                else:
                    rel2pairs[rel] = [pair]

        nn_losser = nn.BCELoss()
        kg_losser = nn.NLLLoss()

        num = 0
        batch = defaultdict(list)
        for i in range(iters):
            sample_space = [(pair, info) for pair, info in self.train_data.items() if '[NA]' not in info.relations]
            sample_space += sample([(pair, info) for pair, info in self.train_data.items() if '[NA]' in info.relations], len(sample_space) // len(self.rels))
            shuffle(sample_space)

            total_nn_loss = 0.0
            total_kg_loss = 0.0
            for pair, info in sample_space:
                num += 1
                # for nn
                batch['h_idx'].append(self.ent2idx[pair[0]])
                batch['t_idx'].append(self.ent2idx[pair[1]])
                examples = [choice(info.examples) for _ in range(20)]
                batch['X'].append([ex.indices for ex in examples])
                batch['pos_pos_1'].append([ex.pos_pos_1 for ex in examples])
                batch['pos_pos_2'].append([ex.pos_pos_2 for ex in examples])
                batch['neg_pos_1'].append([ex.neg_pos_1 for ex in examples])
                batch['neg_pos_2'].append([ex.neg_pos_2 for ex in examples])
                batch['entity_pos_1'].append([ex.entity_pos_1 for ex in examples])
                batch['entity_pos_2'].append([ex.entity_pos_2 for ex in examples])
                batch['target'].append(
                    [0] * len(self.rel2idx)
                )
                for rel in info.relations:
                    batch['target'][-1][self.rel2idx[rel]] = 1.0 / len(info.relations)
                # for kg
                r = choice(info.relations)
                all_h_indices = [self.ent2idx[t[0]] for t in rel2pairs[r]]
                all_t_indices = [self.ent2idx[t[1]] for t in rel2pairs[r]]
                batch['r_idx'].append(self.rel2idx[r])
                batch['h_indices'].append([choice(all_h_indices) for _ in range(20)])
                batch['t_indices'].append([choice(all_t_indices) for _ in range(20)])

                if num == batch_size:
                    nn_output = self.model(
                        X=torch.LongTensor(batch['X']).cuda() if CUDA else torch.LongTensor(batch['X']),
                        positions=[
                            torch.LongTensor(batch['pos_pos_1']).cuda() if CUDA else torch.LongTensor(batch['pos_pos_1']),
                            torch.LongTensor(batch['pos_pos_2']).cuda() if CUDA else torch.LongTensor(batch['pos_pos_2']),
                            torch.LongTensor(batch['neg_pos_1']).cuda() if CUDA else torch.LongTensor(batch['neg_pos_1']),
                            torch.LongTensor(batch['neg_pos_2']).cuda() if CUDA else torch.LongTensor(batch['neg_pos_2']),
                            torch.LongTensor(batch['entity_pos_1']).cuda() if CUDA else torch.LongTensor(batch['entity_pos_1']),
                            torch.LongTensor(batch['entity_pos_2']).cuda() if CUDA else torch.LongTensor(batch['entity_pos_2']),
                        ],
                        h_idx=torch.LongTensor(batch['h_idx']).cuda() if CUDA else torch.LongTensor(batch['h_idx']),
                        t_idx=torch.LongTensor(batch['t_idx']).cuda() if CUDA else torch.LongTensor(batch['t_idx']),
                        r_idx=None,
                        h_indices=None,
                        t_indices=None,
                        nn=True,
                    )

                    nn_loss = nn_losser(nn_output, torch.FloatTensor(batch['target']).cuda() if CUDA else torch.FloatTensor(batch['target']))
                    self.nn_optimizer.zero_grad()
                    nn_loss.backward()
                    self.nn_optimizer.step()
                    total_nn_loss += nn_loss.item()

                    nn_output = None
                    nn_loss = None
                    if CUDA:
                        torch.cuda.empty_cache()

                    pr, ph, pt = self.model(
                        X=None,
                        positions=None,
                        h_idx=torch.LongTensor(batch['h_idx']).cuda() if CUDA else torch.LongTensor(batch['h_idx']),
                        t_idx=torch.LongTensor(batch['t_idx']).cuda() if CUDA else torch.LongTensor(batch['t_idx']),
                        r_idx=torch.LongTensor(batch['r_idx']).cuda() if CUDA else torch.LongTensor(batch['r_idx']),
                        h_indices=torch.LongTensor(batch['h_indices']).cuda() if CUDA else torch.LongTensor(batch['h_indices']),
                        t_indices=torch.LongTensor(batch['t_indices']).cuda() if CUDA else torch.LongTensor(batch['t_indices']),
                        nn=False,
                    )

                    kg_loss = (
                        kg_losser(pr, torch.LongTensor(batch['r_idx']).cuda() if CUDA else torch.LongTensor(batch['r_idx'])) +
                        kg_losser(ph, torch.LongTensor(batch['h_idx']).cuda() if CUDA else torch.LongTensor(batch['h_idx'])) +
                        kg_losser(pt, torch.LongTensor(batch['t_idx']).cuda() if CUDA else torch.LongTensor(batch['t_idx']))
                    )
                    self.kg_optimizer.zero_grad()
                    kg_loss.backward()
                    self.kg_optimizer.step()
                    total_kg_loss += kg_loss.item()

                    num = 0
                    batch = defaultdict(list)

            if i % 10 == 0:
                print("Iter {} nn loss {} kg loss {}".format(i, total_nn_loss, total_kg_loss))
                torch.save(self.model.state_dict(), persist_path)

                predict_scores, facts, predict_rels, pairs = self.predict('tmp')
                precision, recall, _ = precision_recall_curve(facts, predict_scores, pos_label=1)
                self.print_curve(precision, recall)
                self.model.train()


    def predict(self, result_filename, topk=1):
        torch.no_grad()
        self.model.eval()

        predict_scores = []
        predict_rels = []
        facts = []
        pairs = []
        count = 0
        to_be_tested = [t for t in self.test_data.items() if '[NA]' not in t[1].relations]
        to_be_tested += [t for t in self.test_data.items() if '[NA]' in t[1].relations]

        for pair, info in to_be_tested:
            h_idx = self.ent2idx[pair[0]]
            t_idx = self.ent2idx[pair[1]]
            output = self.model(
                X=torch.LongTensor([[ex.indices for ex in info.examples]]).cuda() if CUDA else torch.LongTensor([[ex.indices for ex in info.examples]]),
                positions=[
                    torch.LongTensor([[ex.pos_pos_1 for ex in info.examples]]).cuda() if CUDA else torch.LongTensor([[ex.pos_pos_1 for ex in info.examples]]),
                    torch.LongTensor([[ex.pos_pos_2 for ex in info.examples]]).cuda() if CUDA else torch.LongTensor([[ex.pos_pos_2 for ex in info.examples]]),
                    torch.LongTensor([[ex.neg_pos_1 for ex in info.examples]]).cuda() if CUDA else torch.LongTensor([[ex.neg_pos_1 for ex in info.examples]]),
                    torch.LongTensor([[ex.neg_pos_2 for ex in info.examples]]).cuda() if CUDA else torch.LongTensor([[ex.neg_pos_2 for ex in info.examples]]),
                    torch.LongTensor([[ex.entity_pos_1 for ex in info.examples]]).cuda() if CUDA else torch.LongTensor([[ex.entity_pos_1 for ex in info.examples]]),
                    torch.LongTensor([[ex.entity_pos_2 for ex in info.examples]]).cuda() if CUDA else torch.LongTensor([[ex.entity_pos_2 for ex in info.examples]]),
                ],
                h_idx=torch.LongTensor([h_idx]).cuda() if CUDA else torch.LongTensor([h_idx]),
                t_idx=torch.LongTensor([t_idx]).cuda() if CUDA else torch.LongTensor([t_idx]),
                r_idx=None,
                h_indices=None,
                t_indices=None,
                nn=True,
            )
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

            count += 1
            if count % 100 == 0:
                print("Predicted {}".format(count))
        assert len(predict_scores) == len(facts) == len(predict_rels) == len(pairs)
        with open(result_filename, 'w') as file:
            for predict_score, fact, predict_rel, pair in zip(predict_scores, facts, predict_rels, pairs):
                file.write("{},{},{},{},{}\n".format(predict_score, fact, predict_rel, pair[0], pair[1]))
        return predict_scores, facts, predict_rels, pairs

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
