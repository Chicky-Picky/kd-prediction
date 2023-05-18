# Electrostatic energy for chain-like molecules prediction
_This directory provides examples for electrostatic energy of chain-like molecules prediction_

To run the examples after cloning our git repository, please, make sure you are in `kd-prediction/PPINN` directory.

First, you need to generate chain-like molecules with our script:
`python utils/generate.py`.

Generated data will be stored in `data` directory: both csv-files with atom coordinates and charges and a csv-file with electrostatic energies computed.

Then you may want to train our models from example configs as follows:
- `EdgeConvNodeGATModel`: `python train.py -c ../electrostatic/configs/config_graph_EdgeConvNodeGATModel.json --run-dir EdgeConvNodeGATModel_example`

You will find the results of the training in `experiments` directory: checkpoints and logs for both models will be stored in `experiments/WormLikeChainGraphDataset_Net/checkpoint` and `experiments/WormLikeChainGraphDataset_Net/log` directories, respectively.

To check training process via plotting train and validation losses and metrics, please, first make sure you are in `kd-prediction/PPINN` directory and then run our script as follows:
- `python train_plotter.py`

You will find the resulting pdf-file `TrainResults.pdf` in `experiments/WormLikeChainGraphDataset_Net/log/EdgeConvNodeGATModel_example/` directory.

To check training results and test statistics via plotting, please, first make sure you are in `kd-prediction/PPINN` directory and then run our scripts as follows:
- `python test.py`
- `python test_plotter.py`

You will find the resulting png-files in `experiments/WormLikeChainGraphDataset_Net/test_log/EdgeConvNodeGATModel_example/` directory.
