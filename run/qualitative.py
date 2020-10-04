import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from data.alt_dataset import DataSet as AltDataSet
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

dataset = AltDataSet()
_, test_data = dataset.load_from_file(1)

truths = {}
for pair, info in test_data.items():
    for r in info.relations:
        if r in truths:
            truths[r] += 1
        else:
            truths[r] = 1
print(truths)


label = 'han'
with open("result_{}_{}_{}.csv".format(label, 'top1', '0_0'), 'r') as file:
    scores = []
    facts = []
    predictions = {}
    prediction_counts = {}
    pairs = []
    for line in file:
        score, fact, rel, pair1, pair2 = line[:-1].split(',')
        scores.append(float(score))
        facts.append(int(fact))
        predictions[(pair1, pair2)] = rel
        pairs.append((pair1, pair2))
        if rel in prediction_counts:
            prediction_counts[rel] += 1
        else:
            prediction_counts[rel] = 1
    precision, recall, thresholds = precision_recall_curve(facts, scores, pos_label=1)
    t1 = thresholds[len(thresholds) // 4 * 3]

    print(prediction_counts)
