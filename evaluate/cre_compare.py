from collections import Counter
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

from cre.cnn_transe_runner import CnnTranseRunner
from cre.cnn_complex_runner import CnnComplexRunner
from cre.lstm_transe_runner import LstmTranseRunner
from cre.lstm_complex_runner import LstmComplexRunner
from cre.transformer_transe_runner import TransformerTranseRunner
from cre.transformer_complex_runner import TransformerComplexRunner

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

cnn_transe_runner_supplier = lambda: CnnTranseRunner(load_model='../trained_models/cnn_transe.mod')
cnn_complex_runner_supplier = lambda: CnnComplexRunner(load_model='../trained_models/cnn_complex.mod')
lstm_transe_runner_supplier = lambda: LstmTranseRunner(load_model='../trained_models/lstm_transe.mod')
lstm_complex_runner_supplier = lambda: LstmComplexRunner(load_model='../trained_models/lstm_complex.mod')
transformer_transe_runner_supplier = lambda: TransformerTranseRunner(load_model='../trained_models/transformer_transe.mod')
transformer_complex_runner_supplier = lambda: TransformerComplexRunner(load_model='../trained_models/transformer_complex.mod')

for supplier, label in [
        (cnn_transe_runner_supplier, 'CNN+TransE'),
        (cnn_complex_runner_supplier, 'CNN+ComplEx'),
        (lstm_transe_runner_supplier, 'LSTM+TransE'),
        (lstm_complex_runner_supplier, 'LSTM+ComplEx'),
        (transformer_transe_runner_supplier, 'Transformer+TransE'),
        (transformer_complex_runner_supplier, 'Transformer+ComplEx'),
]:
    runner = supplier()
    print('Predicting ' + label)
    predict_scores, facts, predict_rels, pairs = runner.predict()
    print(Counter(predict_rels))
    precision, recall, thresholds = precision_recall_curve(facts, predict_scores, pos_label=1)
    ax1.plot(recall, precision, label=label)

    predict_scores, facts, predict_rels, pairs = runner.predict(topk=3)
    precision, recall, thresholds = precision_recall_curve(facts, predict_scores, pos_label=1)
    ax2.plot(recall, precision, label=label)

    predict_scores, facts, predict_rels, pairs = runner.predict(topk=238)
    precision, recall, thresholds = precision_recall_curve(facts, predict_scores, pos_label=1)
    ax3.plot(recall, precision, label=label)

    runner = None

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
