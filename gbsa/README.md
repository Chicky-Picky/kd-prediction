# GBSA energy for real molecules prediction
_This directory provides examples for GBSA energy of real molecules prediction_

Prior to training models, you need to prepare the data and calculate GBSA energy of your complexes with AMBER package. On data preparation, please, refer to the [data_preparation](https://github.com/Chicky-Picky/kd-prediction/tree/main/data_preparation#data-preparation) section.

To run the examples after cloning our git repository, please, make sure you are in `kd-prediction/PPINN` directory.

You may want to train our model from example config as follows:
- `EdgeConvNodeGATModel`: `python train.py -c ../gbsa/configs/config_graph_EdgeConvNodeGATModel.json --run-dir EdgeConvNodeGATModel_example`

You will find the results of the training in `experiments` directory: checkpoints and logs will be stored in `experiments/GbsaDgGraphDataset_Net/checkpoint` and `experiments/GbsaDgGraphDataset_Net/log` directories, respectively.

To check training process via plotting train and validation losses and metrics, please, first make sure you are in `kd-prediction/PPINN` directory and then run our script as follows:
- `python train_plotter.py`

You will find the resulting pdf-file `TrainResults.pdf` in `experiments/GbsaDgGraphDataset_Net/log/EdgeConvNodeGATModel_example/` directory.

To check training results and test statistics via plotting, please, first make sure you are in `kd-prediction/PPINN` directory and then run our scripts as follows:
- `python test.py`
- `python test_plotter.py`

You will find the resulting png-files in `experiments/GbsaDgGraphDataset_Net/test_log/EdgeConvNodeGATModel_example/` directory.
