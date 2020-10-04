from collections import Counter
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

from weston.runner import Runner as WestonRunner
from han.runner import Runner as HanRunner
from cre.cnn_transe_runner import CnnTranseRunner

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

weston_runner = WestonRunner(load_m2r_from='../trained_models/weston_m2r.mod', load_transe_from='../trained_models/weston_transe.mod')
weston_runner.label = 'Weston'
han_runner = HanRunner(load_model='../trained_models/han.mod')
han_runner.label = 'Han'
cnn_transe_runner = CnnTranseRunner(load_model='../trained_models/cnn_transe.mod')
cnn_transe_runner.label = 'Core'

for runner in [weston_runner, han_runner, cnn_transe_runner]:
    print('Predicting ' + runner.label)
    predict_scores, facts, predict_rels, pairs = runner.predict()
    print(Counter(predict_rels))
    precision, recall, thresholds = precision_recall_curve(facts, predict_scores, pos_label=1)
    ax1.plot(recall, precision, label=runner.label)

    predict_scores, facts, predict_rels, pairs = runner.predict(topk=3)
    precision, recall, thresholds = precision_recall_curve(facts, predict_scores, pos_label=1)
    ax2.plot(recall, precision, label=runner.label)

    predict_scores, facts, predict_rels, pairs = runner.predict(topk=238)
    precision, recall, thresholds = precision_recall_curve(facts, predict_scores, pos_label=1)
    ax3.plot(recall, precision, label=runner.label)

ax1.set_title('Top-1')
ax2.set_title('Top-3')
ax3.set_title('All')
ax1.set_xlabel('recall')
ax1.set_ylabel('precision')
ax2.set_xlabel('recall')
ax2.set_ylabel('precision')
ax3.set_xlabel('recall')
ax3.set_ylabel('precision')
ax1.legend()
ax2.legend()
ax3.legend()
plt.show()
