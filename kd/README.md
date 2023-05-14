# log(kd) prediction
_This directory provides examples for log(kd) prediction_

There are two ways to predict log(kd) within our workflow:

1) using EdgeConvNodeGAT pretrained (on GBSA dataset) embeddings
2) using ProteinMPNN C-alpha model embeddings

If you intend to use our embeddings, please, follow the instructions:
1) make sure you are in `kd-prediction/kd-prediction/PPINN` directory and run `python embeddings_extraction.py`
2) make sure you are in `kd-prediction/kd-prediction/kd/EdgeConvNodeGAT` directory and run `python add_embeddings_to_interface.py`
3) make sure you are in `kd-prediction/kd-prediction/PPINN` directory and run `python train.py -c ../kd/EdgeConvNodeGAT/configs/config_graph_EdgeConvNodeGATModel.json --run-dir EdgeConvNodeGATModel_embeddings_example`

If you intend to use ProteinMPNN C-alpha model embeddings, please, follow the instructions:
1) clone ProteinMPNN repository
2) run `python ProteinMPNN/helper_scripts/parse_multiple_chains.py` with relevant path arguments and `--ca_only` flag
3) make sure you are in `kd-prediction/kd-prediction/kd/ProteinMPNN` directory and run `python add_embeddings_to_interface.py`
4) make sure you are in `kd-prediction/kd-prediction/PPINN` directory and run `python train.py -c ../kd/ProteinMPNN/configs/config_graph_EdgeConvNodeGATModel.json --run-dir ProteinMPNN_embeddings_example`
