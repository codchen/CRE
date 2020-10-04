from datetime import datetime
from random import sample
from collections import defaultdict

from nltk import word_tokenize, pos_tag
from sklearn.model_selection import train_test_split

class Example:
    def __init__(self, sentence, head, tail):
        if not sentence or not head or not tail:
            return
        self.h_idx = sentence.find(head)
        self.t_idx = sentence.find(tail)
        if self.h_idx < self.t_idx:
            self.pre = sentence[:self.h_idx]
            self.mid = sentence[self.h_idx+len(head):self.t_idx]
            self.post = sentence[self.t_idx+len(tail):]
        else:
            self.pre = sentence[:self.t_idx]
            self.mid = sentence[self.t_idx+len(tail):self.h_idx]
            self.post = sentence[self.h_idx+len(head):]
        self.sentence = sentence
        self.head = head
        self.tail = tail
        self.pair = (head, tail)


class PairData:
    def __init__(self, pair, examples, relations):
        self.pair = pair
        self.examples = examples
        self.relations = relations


class DataSet:
    def __init__(self):
        og_mids_to_words = {}
        with open('../data/mids_to_words.csv', 'r') as file:
            for line in file:
                mid, word = line[:-1].split(',', 1)
                og_mids_to_words[mid] = word
        with open('../data/alt_mids_to_words_unfiltered.csv', 'r') as file:
            for line in file:
                mid, word = line[:-1].split(',', 1)
                og_mids_to_words[mid] = word
        words_to_mids = { word: mid for mid, word in og_mids_to_words.items() }
        mids_to_words = { mid: word for word, mid in words_to_mids.items() }
        # everything below is in word
        fb_all_entities = set([])
        fb_all_relations = set([])
        fb_pairs_to_relations = {}
        with open('../data/relations_alt.csv', 'r') as file:
            for line in file:
                e1, r, e2 = line[:-1].split(',')
                if e1 not in og_mids_to_words or e2 not in og_mids_to_words:
                    continue
                w1 = og_mids_to_words[e1]
                w2 = og_mids_to_words[e2]
                fb_all_entities.add(w1)
                fb_all_entities.add(w2)
                fb_all_relations.add(r)

                entity_pair = (w1, w2)
                if entity_pair in fb_pairs_to_relations:
                    fb_pairs_to_relations[entity_pair].append(r)
                else:
                    fb_pairs_to_relations[entity_pair] = [r]

        self.examples = []
        self.all_entities = set([])
        self.all_pairs = set([])
        with open('../data/nyt_all_simplified.csv', 'r') as file:
            for line in file:
                w1, w2, sentence = line[:-1].split('|')
                if w1 not in fb_all_entities or w2 not in fb_all_entities or len(sentence.split(' ')) > 300:
                    continue
                example = Example(sentence=sentence, head=w1, tail=w2)
                self.examples.append(example)
                self.all_entities.add(w1)
                self.all_entities.add(w2)
                self.all_pairs.add(example.pair)

        self.pairs_to_relations = {}
        self.positive_pairs = set([])
        self.negative_pairs = set([])
        self.all_relations = set([])
        for pair in self.all_pairs:
            if pair in fb_pairs_to_relations:
                self.pairs_to_relations[pair] = fb_pairs_to_relations[pair]
                self.positive_pairs.add(pair)
            else:
                self.pairs_to_relations[pair] = ['[NA]']
                self.negative_pairs.add(pair)
            self.all_relations.update(self.pairs_to_relations[pair])
        self.examples_by_pairs = self.get_examples_by_pairs()

        print(
            "Dataset Summary: \n"
            f"Number of sentences: {len(self.examples)} \n"
            f"Number of entities: {len(self.all_entities)} \n"
            f"Number of entity pairs: {len(self.all_pairs)} including {len(self.positive_pairs)} positive and {len(self.negative_pairs)} negative \n"
            f"Number of relations: {len(self.all_relations)} \n"
        )
        self.sentence_summary()

    def divide_into_subsets(self, test_ratio=0.2, num=3):
        positive_size = len(self.positive_pairs) // num
        negative_size = len(self.negative_pairs) // num
        for i in range(num - 1):
            positive_pairs = set(sample(self.positive_pairs, positive_size))
            negative_pairs = set(sample(self.negative_pairs, negative_size))

            train_positive_pairs, test_positive_pairs = train_test_split(list(positive_pairs), test_size=test_ratio)
            train_negative_pairs, test_negative_pairs = train_test_split(list(negative_pairs), test_size=test_ratio)
            test_negative_pairs = sample(test_negative_pairs, len(test_positive_pairs) // len(self.all_relations))

            with open("../data/alt_train_{}".format(i), 'w') as file:
                for pair in train_positive_pairs + train_negative_pairs:
                    file.write("{},{}\n".format(pair[0], pair[1]))
            with open("../data/alt_test_{}".format(i), 'w') as file:
                for pair in test_positive_pairs + test_negative_pairs:
                    file.write("{},{}\n".format(pair[0], pair[1]))

            self.positive_pairs -= positive_pairs
            self.negative_pairs -= negative_pairs
        train_positive_pairs, test_positive_pairs = train_test_split(list(self.positive_pairs), test_size=test_ratio)
        train_negative_pairs, test_negative_pairs = train_test_split(list(self.negative_pairs), test_size=test_ratio)
        test_negative_pairs = sample(test_negative_pairs, len(test_positive_pairs) // len(self.all_relations))

        with open("../data/alt_train_{}".format(num - 1), 'w') as file:
            for pair in train_positive_pairs + train_negative_pairs:
                file.write("{},{}\n".format(pair[0], pair[1]))
        with open("../data/alt_test_{}".format(num - 1), 'w') as file:
            for pair in test_positive_pairs + test_negative_pairs:
                file.write("{},{}\n".format(pair[0], pair[1]))

    def load_from_file(self, num):
        train_pairs = []
        test_pairs = []
        with open("../data/alt_train_{}".format(num), 'r') as file:
            for line in file:
                t1, t2 = line[:-1].split(',')
                train_pairs.append((t1, t2))
        with open("../data/alt_test_{}".format(num), 'r') as file:
            for line in file:
                t1, t2 = line[:-1].split(',')
                test_pairs.append((t1, t2))
        train_pair_data = {
            pair: PairData(pair, self.examples_by_pairs[pair], sorted(self.pairs_to_relations[pair]))
            for pair in train_pairs
        }
        test_pair_data = {
            pair: PairData(pair, self.examples_by_pairs[pair], sorted(self.pairs_to_relations[pair]))
            for pair in test_pairs if pair in self.examples_by_pairs and pair in self.pairs_to_relations
        }
        return train_pair_data, test_pair_data
    
    def featurize_all(self):
        self.vocab = set([])
        for example in self.examples:
            example.feature = self.featurize(example)
            self.vocab.update(example.feature)
        print(f"Size of vocabulary: {len(self.vocab)}")

    def featurize(self, ex):
        return ex.pre.split(' ') + ex.mid.split(' ') + ex.post.split(' ')

    def split(self, ratio=0.25, load_from_file=True):
        if load_from_file:
            test_pairs = []
            with open('../data/test_pairs.csv', 'r') as file:
                for line in file:
                    t1, t2 = line[:-1].split(',')
                    test_pairs.append((t1, t2))
            train_pairs = list(self.all_pairs - set(test_pairs))
        else:
            train_positive_pairs, test_positive_pairs = train_test_split(list(self.positive_pairs), test_size=ratio)
            train_negative_pairs, test_negative_pairs = train_test_split(list(self.negative_pairs), test_size=ratio)
            train_pairs = train_positive_pairs + train_negative_pairs
            test_pairs = test_positive_pairs + test_negative_pairs
        train_pair_data = {
            pair: PairData(pair, self.examples_by_pairs[pair], sorted(self.pairs_to_relations[pair]))
            for pair in train_pairs
        }
        test_pair_data = {
            pair: PairData(pair, self.examples_by_pairs[pair], sorted(self.pairs_to_relations[pair]))
            for pair in test_pairs if pair in self.examples_by_pairs and pair in self.pairs_to_relations
        }
        return train_pair_data, test_pair_data

    def get_examples_by_pairs(self):
        result = {}
        for example in self.examples:
            if example.pair in result:
                result[example.pair].append(example)
            else:
                result[example.pair] = [example]
        return result

    def sentence_summary(self):
        num_pos_sent = len([sent for sent in self.examples if sent.pair in self.positive_pairs])
        num_neg_sent = len([sent for sent in self.examples if sent.pair in self.negative_pairs])
        print(
            f"Number of sentences for positive pairs: {num_pos_sent}\n"
            f"Number of sentences for negative pairs: {num_neg_sent}\n"
        )
