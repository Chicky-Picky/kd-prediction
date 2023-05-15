# $`\ln(K_d)`$ prediction
_This directory provides examples for $`\ln(K_d)`$ prediction_

Prior to training models, you need to prepare the data. For this purpose, please, refer to the [data_preparation](https://github.com/Chicky-Picky/kd-prediction/tree/main/data_preparation#data-preparation) section.

There are two ways to predict $`\ln(K_d)`$ within our workflow:

- using EdgeConvNodeGAT pretrained (on GBSA dataset) embeddings
- using ProteinMPNN $`C_\alpha`$ only model embeddings

If you intend to use our embeddings, please, follow the instructions:
1) make sure you are in `kd-prediction/PPINN` directory and run `python embeddings_extraction.py` to extract pretrained embeddings
2) make sure you are in `kd-prediction/kd/EdgeConvNodeGAT` directory and run `python add_embeddings_to_interface.py` to store embeddings in the dataset
3) make sure you are in `kd-prediction/PPINN` directory and run `python train.py -c ../kd/EdgeConvNodeGAT/configs/config_graph_EdgeConvNodeGATModel.json --run-dir EdgeConvNodeGATModel_embeddings_example`

You will find the results of the training in `experiments` directory: checkpoints and logs will be stored in `experiments/GbsaKdGraphDataset_Net/checkpoint` and `experiments/GbsaKdGraphDataset_Net/log` directories, respectively.

If you intend to use ProteinMPNN $`C_\alpha`$ only model embeddings, please, follow the instructions:
1) clone ProteinMPNN repository
2) run `python ProteinMPNN/helper_scripts/parse_multiple_chains.py` with relevant path arguments and `--ca_only` flag to parse your pdb-files
3) make sure you are in `kd-prediction/kd-prediction/kd/ProteinMPNN` directory and run `python add_embeddings_to_interface.py` to store embeddings in the dataset
4) make sure you are in `kd-prediction/kd-prediction/PPINN` directory and run `python train.py -c ../kd/ProteinMPNN/configs/config_graph_EdgeConvNodeGATModel.json --run-dir ProteinMPNN_embeddings_example`

You will find the results of the training in `experiments` directory: checkpoints and logs will be stored in `experiments/GbsaKdGraphDataset_Net/checkpoint` and `experiments/GbsaKdGraphDataset_Net/log` directories, respectively.
