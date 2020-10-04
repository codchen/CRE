## Setup
First, create a virtual environment with conda: `conda create -n cre python=3.7` and activate it: `conda activate cre`.

Then, install all dependencies with `pip install -r requirements.txt`

## Data
Download data from , unzip it, and move all files under CRE/data

## Training & Prediction
Open `run/train_and_predict.py` and scroll to the bottom. Comment out any models that you don't want to run for (in other words, leave only the models you want to run uncommented). For example, if you want to run `cnn+TransE` and `transformer+ComplEx`, the bottom of `run/train_and_predict.py` should look like:
```python
# train_and_predict(weston_runner_supplier, weston_runner_trainer, 'weston')
# train_and_predict(han_runner_supplier, han_runner_trainer, 'han')
train_and_predict(cnn_transe_runner_supplier, cnn_transe_runner_trainer, 'cnn_transe')
# train_and_predict(cnn_complex_runner_supplier, cnn_complex_runner_trainer, 'cnn_complex')
# train_and_predict(lstm_transe_runner_supplier, lstm_transe_runner_trainer, 'lstm_transe')
# train_and_predict(lstm_complex_runner_supplier, lstm_complex_runner_trainer, 'lstm_complex')
# train_and_predict(transformer_transe_runner_supplier, transformer_transe_runner_trainer, 'transformer_transe')
train_and_predict(transformer_complex_runner_supplier, transformer_complex_runner_trainer, 'transformer_complex')
```
Then, do the following to train & predict:
```bash
cd run
python train_and_predict.py
```
All trained models will be stored in `trained_models/`, and all prediction results will be stored in `run/`

## Visualization
Once prediction results are made, you can use helper functions in `run/visualize.py` to view plots for the results. Specifically, `compare_results_across_datasets` plot results for top-1, top-3, and top-10, within each of which are the prediction results of your selected model on 3 different datasets. `compare_results_within_dataset`, on the other hand, plot results for top-1, top-3, and top-10 of your selected model on the same dataset, but across 3 different training runs. Suppose you want to compare across datasets for Weston's model, and compare across training runs for `cnn+TransE`, the bottom of `run/visualize.py` should look like:
```python
compare_results_across_datasets('weston')
compare_results_within_dataset('cnn_transe')
```
Then, do the following to show plot:
```bash
cd run
python visualize.py
```
