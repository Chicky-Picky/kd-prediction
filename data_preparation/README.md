# Data preparation
_This directory provides examples for data preparation from raw pdb-files for both GBSA and $`K_d`$ prediction_

To run the examples after cloning our git repository, please, make sure you are in `kd-prediction/data_preparation` directory.

You may want to run our utility scripts as follows:
- To extract interfaces from corrected pdb-files into pdb-files: `python interface_extraction.py`
- To create csv-files with atom cooordinates and one-hot-encoding features: `python add_features_to_interface.py`

`clean_pdb/` directory contains physically corrected pdb-files (please, refer to **GBSA free energy prediction** section [Methods](https://github.com/Chicky-Picky/kd-prediction/#methods)).

`interface_pdb/` directory contains pdb-files with interfaces only, after running `interface_extraction.py`.

`interface_with_feature/` directory contains csv-files with interface atom cooordinates and one-hot-encoding features for GBSA prediction training, after running `add_features_to_interface.py`.
