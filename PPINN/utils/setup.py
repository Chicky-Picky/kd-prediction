from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from pathlib import Path
from utils.util import write_json, wrapped_partial
from typing import Dict
import os
import logging
import logging.config
import numpy as np


class SetupRun:
    def __init__(self,
                 config: Dict,
                 checkpoint_dir: str,
                 resume=None,
                 ):
        """
        class to parse configuration json file. Handles hyperparameters for training,
        initializations of modules, checkpoint saving and logging module.
        :param config: Dict containing configurations, hyperparameters for training.
                       contents of `config.json` file for example.
        :param resume: String, path to the checkpoint being loaded.
        """
        self.config = config
        self.resume = resume

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # save config file to the checkpoint dir
        write_json(self.config, self.checkpoint_dir / 'config.json')

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_funct(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return wrapped_partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        if name in self.__dict__["config"]:
            return self.__dict__["config"][name]
        elif name in self.__dict__["config"]["metrics"]:
            return self.__dict__["config"]["metrics"][name]
        elif name in self.__dict__["config"]["dataset_transforms"]:
            return self.__dict__["config"]["dataset_transforms"][name]
        else:
            return self.__dict__[name]


class SetupLogger:
    def __init__(self,
                 config: Dict,
                 log_dir: str = "."):

        os.makedirs(log_dir, exist_ok=True)
        config["handlers"]["info_file_handler"]["filename"] = os.path.join(log_dir, "log.log")

        self.config = config
        self.log_dir = log_dir

        logging.config.dictConfig(config)

    def __call__(self,
                 name: str,
                 log_level: int = logging.DEBUG):
        logger = logging.getLogger(name)
        logger.setLevel(log_level)
        return logger

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        if name in self.__dict__["config"]:
            return self.__dict__["config"][name]
        else:
            return self.__dict__[name]


def setup_data_samplers(dataset_size, validation_split, shuffle, random_seed):
    indices = list(range(dataset_size))
    validation_split = int(np.floor(validation_split * dataset_size))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[validation_split:], indices[:validation_split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    return train_sampler, valid_sampler


def setup_data_loaders(
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
        validation_split: float,
        num_workers: int,
        random_seed: int = 0):
    init_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers
    }
    train_sampler, validation_sampler = setup_data_samplers(len(dataset), validation_split, shuffle, random_seed)
    train_loader = DataLoader(dataset=dataset, sampler=train_sampler, **init_kwargs)
    validation_loader = DataLoader(dataset=dataset, sampler=validation_sampler, **init_kwargs)

    return train_loader, validation_loader


def setup_split_data_loaders(
        dataset_train: Dataset,
        dataset_valid: Dataset,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        **kwargs):

    init_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers
    }

    if shuffle:
        dataset_train.shuffle()
        dataset_valid.shuffle()

    train_loader = DataLoader(dataset=dataset_train, **init_kwargs)
    validation_loader = DataLoader(dataset=dataset_valid, **init_kwargs)

    return train_loader, validation_loader
