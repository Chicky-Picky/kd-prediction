from torch_geometric.data import Data, Dataset
from glob import glob
from tqdm import tqdm
from scipy.spatial.distance import cdist
import torch
import pandas as pd
import os


class BaseCsvDataset(Dataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 raw_dirname="raw",
                 processed_dirname='processed'
                 ):
        """
        :param root: where dataset should be stored. This folder is split into
                     raw_dir (original datset) and processed_dir (processed data (graph))
        :param transform:
        :param pre_transform:
        :param pre_filter:
        :param raw_dirname: override the 'raw' name in base class
                            self.raw_dir = os.path.join(self.root, 'raw')
        :param processed_dirname: override the 'processed' name in base class
                                  self.processed_dir = os.path.join(self.root, 'processed')
        """
        self.raw_dirname = raw_dirname
        self.processed_dirname = processed_dirname
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.raw_dirname)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.processed_dirname)

    @property
    def raw_file_names(self):
        r"""The name of the files in the :obj:`self.raw_dir` folder that must
        be present in order to skip downloading."""
        mol_csvs = [os.path.basename(file) for file in
                    glob(os.path.join(self.raw_dir, "*.csv"))]
        mol_csvs.sort()
        return mol_csvs

    @property
    def processed_file_names(self):
        r"""The name of the files in the :obj:`self.processed_dir` folder that
        must be present in order to skip processing."""
        mols_pts = [os.path.basename(file) for file in
                    glob(os.path.join(self.processed_dir, "*.pt"))
                    if not os.path.basename(file).startswith("pre")
                    ]
        mols_pts.sort()
        return mols_pts

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        r"""Gets the data object at index :obj:`idx`."""
        NotImplemented

    def process(self):
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
        for raw_file_path in tqdm(self.raw_paths):
            mol_df = pd.read_csv(raw_file_path)
            # get node features
            node_features = self._get_node_features(mol_df)
            # get edge features
            edge_features = self._get_edge_features(mol_df)
            # get adjacency info
            edge_indexes = self._get_adjacency_info(mol_df)

            # create data object
            data = Data(x=node_features,
                        edge_index=edge_indexes,
                        edge_attr=edge_features,
                        )

            molname = os.path.basename(raw_file_path).split(".")[0]
            torch.save(data, os.path.join(self.processed_dir, f"{molname}.pt"))

    def _get_node_features(self, mol):
        """
        This will return a matrix / 2d array of the shape
        [number of nodes, node features size]
        """
        NotImplemented

    def _get_edge_features(self, mol):
        """
        This will return a matrix / 2d array of the shape
        [number of edges, edge features size]
        """
        NotImplemented

    def _get_adjacency_info(self, mol):
        edges = [[i, j] for i in range(mol.shape[0]) for j in range(mol.shape[0])]
        return torch.tensor(edges, dtype=torch.long).t().contiguous()


class WormLikeChainGraphDataset(BaseCsvDataset):

    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 raw_dirname="raw",
                 processed_dirname='processed'
                 ):
        super().__init__(root, transform, pre_transform, pre_filter, raw_dirname, processed_dirname)

    def get(self, idx):
        molfile_pt = self.processed_file_names[idx]
        data = torch.load(os.path.join(self.processed_dir, molfile_pt))

        el_energy = pd.read_csv(os.path.join(self.root, 'el_energy.csv'))
        molname = os.path.basename(molfile_pt).split(".")[0]
        target = el_energy["el_energy"][el_energy["molname"] == molname].values
        target = torch.from_numpy(target).float()

        return data, target

    def _get_node_features(self, mol):
        """
        This will return a matrix / 2d array of the shape
        [number of nodes, node features size]
        """
        features = torch.tensor(mol[['q']].values, dtype=torch.float)
        return features

    def _get_edge_features(self, mol):
        """
        This will return a matrix / 2d array of the shape
        [number of edges, edge features size]
        """
        xyz = mol[['x', 'y', 'z']].values
        features = cdist(xyz, xyz).reshape(-1, 1)
        return torch.tensor(features, dtype=torch.float)


class GbsaDgGraphDataset(BaseCsvDataset):

    def get(self, idx):
        r"""Gets the data object at index :obj:`idx`."""
        dG = pd.read_csv(os.path.join(self.root, 'dG.csv'))
        molfile_pt = self.processed_file_names[idx]

        target = dG["dG"][dG["complex"] == molfile_pt.split("pdb_")[-1].split(".pt")[0]].values
        target = torch.from_numpy(target).float()

        data = torch.load(os.path.join(self.processed_dir, molfile_pt))

        return data, target

    def _get_node_features(self, mol):
        """
        This will return a matrix / 2d array of the shape
        [number of nodes, node features size]
        """
        onehots_strs = mol['one-hot-atype'].values
        onehots = []
        for string in onehots_strs:
            onehots.append([float(s) for s in string])

        features = torch.tensor(onehots, dtype=torch.float)
        return features

    def _get_edge_features(self, mol):
        """
        This will return a matrix / 2d array of the shape
        [number of edges, edge features size]
        """
        xyz = mol[['x', 'y', 'z']].values
        features = cdist(xyz, xyz).reshape(-1, 1)
        return torch.tensor(features, dtype=torch.float)


class GbsaKdGraphDataset(BaseCsvDataset):

    def get(self, idx):
        r"""Gets the data object at index :obj:`idx`."""
        kd = pd.read_csv(os.path.join(self.root, 'log_kd.csv'))
        molfile_pt = self.processed_file_names[idx]

        target = kd["log_kd"][kd["complex"] == molfile_pt.split("pdb_")[-1].split(".pt")[0]].values
        target = torch.from_numpy(target).float()

        data = torch.load(os.path.join(self.processed_dir, molfile_pt))

        return data, target

    def _get_node_features(self, mol):
        """
        This will return a matrix / 2d array of the shape
        [number of nodes, node features size]
        """
        embeddings_strs = mol['embedding'].values
        embeddings = []
        for string in embeddings_strs:
            embeddings.append([float(s) for s in string.split()])

        features = torch.tensor(embeddings, dtype=torch.float)
        return features

    def _get_edge_features(self, mol):
        """
        This will return a matrix / 2d array of the shape
        [number of edges, edge features size]
        """
        xyz = mol[['x', 'y', 'z']].values
        features = cdist(xyz, xyz).reshape(-1, 1)
        return torch.tensor(features, dtype=torch.float)


if __name__ == '__main__':
    dataset = WormLikeChainGraphDataset(root="../data", raw_dirname="mols")
    # print(dataset[0].edge_index.t())
    # print(dataset[0].edge_attr)
    # print(dataset[0].x)
