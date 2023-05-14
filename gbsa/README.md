# GBSA energy for real molecules prediction
_This directory provides examples for GBSA energy for real molecules prediction_

To run the examples after cloning our git repository, please, make sure you are in `kd-prediction/kd-prediction/PPINN` directory.

You may want to train our models from example config as follows:
- `EdgeConvNodeGATModel`: `python train.py -c ../gbsa/configs/config_graph_EdgeConvNodeGATModel.json --run-dir EdgeConvNodeGATModel_example`