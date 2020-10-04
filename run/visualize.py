import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

def normalize(precision, recall):
    recall_levels = list(map(lambda x: x/1000, range(1000)))
    precision_levels = list(map(lambda r: precision_for_recall(r, recall, precision), recall_levels))
    return precision_levels, recall_levels

def precision_for_recall(recall, recalls_desc, precisions_asc):
    for i, r in enumerate(recalls_desc):
        if r > recall:
            continue
        if i == 0:
            return precisions_asc[i]
        return precisions_asc[i] + (precisions_asc[i] - precisions_asc[i-1]) / (recalls_desc[i-1] - r) * (recall - r)
    return precisions_asc[-1]

def compare_results_across_datasets(labels=[], colors=['r','g','b']):
    _, (ax1, ax2, ax3) = plt.subplots(1, 3)
    mappings = {'top1': ax1, 'top3': ax2, 'top5': ax3}
    
    for c, label in enumerate(labels):
        for topk in ['top1', 'top3', 'top5']:
            p = []
            r = []
            for i in range(3):
                for j in range(3):
                    suffix = "{}_{}".format(i, j)
                    scores = []
                    facts = []
                    with open("result_{}_{}_{}.csv".format(label, topk, suffix), 'r') as file:
                        for line in file:
                            score, fact, _ = line[:-1].split(',', 2)
                            scores.append(float(score))
                            facts.append(int(fact))
                    precision, recall, _ = precision_recall_curve(facts, scores, pos_label=1)
                    precision_levels, recall_levels = normalize(precision, recall)
                    p.append(precision_levels)
                    r.append(recall_levels)
            p = np.asarray(p)
            r = np.asarray(r[0])
            mean_p = np.mean(p, axis=0)
            std_p = np.std(p, axis=0)
            mappings[topk].plot(r, mean_p, label=label, color=colors[c])
            mappings[topk].fill_between(r, (mean_p-std_p), (mean_p+std_p), color=colors[c], alpha=.1)
    for k, v in mappings.items():
        v.set_title(k)
        v.set_xlabel('recall')
        v.set_ylabel('precision')
        v.legend()
    plt.show()

def compare_results_within_dataset(labels='cnn_transe', colors=['r','g','b','y','c','m']):
    _, (ax1, ax2, ax3) = plt.subplots(1, 3)
    mappings = {'top1': ax1, 'top3': ax2, 'top5': ax3}
    
    for c, label in enumerate(labels):
        for topk in ['top1', 'top3', 'top5']:
            p = []
            r = []
            for i in range(1):
                for j in range(3):
                    suffix = "{}_{}".format(2 if label=='transformer_complex' else i, j)
                    scores = []
                    facts = []
                    with open("result_{}_{}_{}.csv".format(label, topk, suffix), 'r') as file:
                        for line in file:
                            score, fact, _ = line[:-1].split(',', 2)
                            scores.append(float(score))
                            facts.append(int(fact))
                    precision, recall, _ = precision_recall_curve(facts, scores, pos_label=1)
                    precision_levels, recall_levels = normalize(precision, recall)
                    p.append(precision_levels)
                    r.append(recall_levels)
            p = np.asarray(p)
            r = np.asarray(r[0])
            mean_p = np.mean(p, axis=0)
            std_p = np.std(p, axis=0)
            mappings[topk].plot(r, mean_p, label=label, color=colors[c])
            mappings[topk].fill_between(r, (mean_p-std_p), (mean_p+std_p), color=colors[c], alpha=.1)
    for k, v in mappings.items():
        v.set_title(k)
        v.set_xlabel('recall')
        v.set_ylabel('precision')
        v.legend()
    plt.show()

compare_results_within_dataset(['cnn_transe','cnn_complex','lstm_transe','lstm_complex','transformer_transe','transformer_complex'])
