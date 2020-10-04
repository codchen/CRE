import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from weston.runner import Runner as WestonRunner
from han.runner import Runner as HanRunner
from cre.cnn_transe_runner import CnnTranseRunner
from cre.cnn_complex_runner import CnnComplexRunner
from cre.lstm_transe_runner import LstmTranseRunner
from cre.lstm_complex_runner import LstmComplexRunner
from cre.transformer_transe_runner import TransformerTranseRunner
from cre.transformer_complex_runner import TransformerComplexRunner
from data.alt_dataset import DataSet

NUM_RUNS = 3

# which_dataset is one of [0, 1, 2]
weston_runner_supplier = lambda which_dataset: WestonRunner(k=100, k_transe=50, gamma=30, use_alt=which_dataset)
def weston_runner_trainer(runner, which_dataset, run_num=None):
    suffix = str(which_dataset) if run_num is None else "{}_{}".format(which_dataset, run_num)
    runner.train_m2r(iters=1, persist_path="../trained_models/weston_m2r_{}.mod".format(suffix))
    runner.train_transe(iters=1, persist_path="../trained_models/weston_transe_{}.mod".format(suffix))

han_runner_supplier = lambda which_dataset: HanRunner(k=50, conv_window=3, hidden_dim=300, use_alt=which_dataset)
def han_runner_trainer(runner, which_dataset, run_num=None):
    suffix = str(which_dataset) if run_num is None else "{}_{}".format(which_dataset, run_num)
    runner.train(iters=1, persist_path="../trained_models/han_{}.mod".format(suffix))

cnn_transe_runner_supplier = lambda which_dataset: CnnTranseRunner(use_alt=which_dataset)
def cnn_transe_runner_trainer(runner, which_dataset, run_num=None):
    suffix = str(which_dataset) if run_num is None else "{}_{}".format(which_dataset, run_num)
    runner.train(epochs=1, persist_path="../trained_models/cnn_transe_{}.mod".format(suffix))

cnn_complex_runner_supplier = lambda which_dataset: CnnComplexRunner(use_alt=which_dataset)
def cnn_complex_runner_trainer(runner, which_dataset, run_num=None):
    suffix = str(which_dataset) if run_num is None else "{}_{}".format(which_dataset, run_num)
    runner.train(epochs=1, persist_path="../trained_models/cnn_complex_{}.mod".format(suffix))

lstm_transe_runner_supplier = lambda which_dataset: LstmTranseRunner(use_alt=which_dataset)
def lstm_transe_runner_trainer(runner, which_dataset, run_num=None):
    suffix = str(which_dataset) if run_num is None else "{}_{}".format(which_dataset, run_num)
    runner.train(epochs=1, persist_path="../trained_models/lstm_transe_{}.mod".format(suffix))

lstm_complex_runner_supplier = lambda which_dataset: LstmComplexRunner(use_alt=which_dataset)
def lstm_complex_runner_trainer(runner, which_dataset, run_num=None):
    suffix = str(which_dataset) if run_num is None else "{}_{}".format(which_dataset, run_num)
    runner.train(epochs=1, persist_path="../trained_models/lstm_complex_{}.mod".format(suffix))

transformer_transe_runner_supplier = lambda which_dataset: TransformerTranseRunner(use_alt=which_dataset)
def transformer_transe_runner_trainer(runner, which_dataset, run_num=None):
    suffix = str(which_dataset) if run_num is None else "{}_{}".format(which_dataset, run_num)
    runner.train(epochs=1, batch_size=10, persist_path="../trained_models/transformer_transe_{}.mod".format(suffix))

transformer_complex_runner_supplier = lambda which_dataset: TransformerComplexRunner(use_alt=which_dataset)
def transformer_complex_runner_trainer(runner, which_dataset, run_num=None):
    suffix = str(which_dataset) if run_num is None else "{}_{}".format(which_dataset, run_num)
    runner.train(epochs=1, batch_size=20, persist_path="../trained_models/transformer_complex_{}.mod".format(suffix))

def train_and_predict(supplier, trainer, result_label):
    for which_dataset in [0, 1, 2]:
        for i in [0, 1, 2]:
            runner = supplier(which_dataset)
            trainer(runner, which_dataset, i)
            runner.predict("result_{}_top1_{}_{}.csv".format(result_label, which_dataset, i), topk=1)
            runner.predict("result_{}_top3_{}_{}.csv".format(result_label, which_dataset, i), topk=3)
            runner.predict("result_{}_top5_{}_{}.csv".format(result_label, which_dataset, i), topk=5)

train_and_predict(weston_runner_supplier, weston_runner_trainer, 'weston')
train_and_predict(han_runner_supplier, han_runner_trainer, 'han')
train_and_predict(cnn_transe_runner_supplier, cnn_transe_runner_trainer, 'cnn_transe')
train_and_predict(cnn_complex_runner_supplier, cnn_complex_runner_trainer, 'cnn_complex')
train_and_predict(lstm_transe_runner_supplier, lstm_transe_runner_trainer, 'lstm_transe')
train_and_predict(lstm_complex_runner_supplier, lstm_complex_runner_trainer, 'lstm_complex')
train_and_predict(transformer_transe_runner_supplier, transformer_transe_runner_trainer, 'transformer_transe')
train_and_predict(transformer_complex_runner_supplier, transformer_complex_runner_trainer, 'transformer_complex')
