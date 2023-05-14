from torch.utils.data import Dataset
import pandas as pd
import torch
import os
from glob import glob


class WormLikeChainDataset(Dataset):
    """Worm-like chain dataset"""

    def __init__(self,
                 path_to_mol_dir: str,
                 path_to_el_energy_csv: str,
                 transform=None
                 ):
        """
        Args:
            path_to_mol_dir: directory with samples - charged molecules.
                             each csv file contains coordinates and charges [x, y, z, q].
            path_to_el_energy_csv: csv file with electrostatic energy values
            transform (callable, optional): Optional transform to be applied  on a sample.
        """
        if transform:
            self.transform = transform

        self.mol_csvs = glob(os.path.join(path_to_mol_dir, "*.csv"))
        self.mol_csvs.sort()
        self.el_energy = pd.read_csv(path_to_el_energy_csv)

    def __len__(self):
        """returns the size of the dataset"""
        return len(self.mol_csvs)

    def __getitem__(self, idx):
        """to support the indexing such that dataset[i] can be used to get iith sample."""
        data = pd.read_csv(self.mol_csvs[idx])
        data = data[['x', 'y', 'z', 'q']].values.flatten()
        molname = os.path.basename(self.mol_csvs[idx]).split(".")[0]
        target = self.el_energy["el_energy"][self.el_energy["molname"] == molname].values
        target = torch.from_numpy(target).float()

        if self.transform:
            data = self.transform(data)

        return data, target
