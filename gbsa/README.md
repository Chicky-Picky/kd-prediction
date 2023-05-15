# GBSA energy for real molecules prediction
_This directory provides examples for GBSA energy of real molecules prediction_

Prior to training models, you need to prepare the data and calculate GBSA energy of your complexes with AMBER package. On data preparation, please, refer to the [data_preparation](https://github.com/Chicky-Picky/kd-prediction/tree/main/data_preparation#data-preparation) section.

To run the examples after cloning our git repository, please, make sure you are in `kd-prediction/PPINN` directory.

You may want to train our model from example config as follows:
- `EdgeConvNodeGATModel`: `python train.py -c ../gbsa/configs/config_graph_EdgeConvNodeGATModel.json --run-dir EdgeConvNodeGATModel_example`

You will find the results of the training in `experiments` directory: checkpoints and logs will be stored in `experiments/GbsaDgGraphDataset_Net/checkpoint` and `experiments/GbsaDgGraphDataset_Net/log` directories, respectively.
