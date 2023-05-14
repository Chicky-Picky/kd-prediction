from utils.setup import setup_data_loaders, setup_split_data_loaders
from utils.setup import SetupRun
from utils.logging_utils import SetupLogger
from utils.util import read_json
from trainer.trainer import Trainer
import dataset as module_dataset
import utils.metric as module_metric
import utils.transform as module_transform
import model.model as module_arch
from utils.visualization import TensorboardWriter

from datetime import datetime
from pathlib import Path
from torchvision import transforms
import argparse
import torch
import torch.nn.functional as F
import os


def run_training(run_setup: SetupRun,
                 logger_setup: SetupLogger,
                 vizualizer_setup,
                 device: str):
    # setup logger
    logger = logger_setup.get_logger("train")

    # setup dataset
    dataset_transforms = [run_setup.init_obj(transform, module_transform) for transform in
                          run_setup['dataset_transforms']]

    assert bool(run_setup["dataset_train"] and run_setup["dataset_valid"] and run_setup["dataset"]) != 1, \
        "Please, specify 1) dataset in config with validation_split parameter or \n 2) dataset_train and dataset_valid"

    if run_setup["dataset_train"] and run_setup["dataset_valid"]:
        dataset_train = run_setup.init_obj('dataset_train', module_dataset,
                                           transform=transforms.Compose(dataset_transforms))
        dataset_valid = run_setup.init_obj('dataset_valid', module_dataset,
                                           transform=transforms.Compose(dataset_transforms))
        logger.info(dataset_train)
        logger.info(dataset_valid)

        # setup data_loader instances
        data_loader, valid_data_loader = setup_split_data_loaders(dataset_train=dataset_train,
                                                                  dataset_valid=dataset_valid,
                                                                  **run_setup["dataloader"]["args"])

    elif run_setup["dataset"]:
        dataset = run_setup.init_obj('dataset', module_dataset, transform=transforms.Compose(dataset_transforms))
        logger.info(dataset)

        # setup data_loader instances
        data_loader, valid_data_loader = setup_data_loaders(dataset=dataset, **run_setup["dataloader"]["args"])

    # setup model architecture, then print to console
    model = run_setup.init_obj('arch', module_arch)
    logger.info(model)

    # setup function handles of loss and metrics
    criterion = run_setup.init_funct("loss", F)
    metrics = [run_setup.init_funct(metric, module_metric) for metric in run_setup['metrics']]

    # setup optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = run_setup.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = run_setup.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    # setup trainer
    trainer = Trainer(model=model,
                      criterion=criterion,
                      metric_ftns=metrics,
                      optimizer=optimizer,
                      run_setup=run_setup,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      vizualizer=vizualizer_setup
                      )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('--run-dir', default=None, type=str,
                        help='name of run directory. If it is None, the current date and time will be used')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default="cpu", type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-l', '--log-config', default="logger_config.json", type=str,
                        help='log config file path (default: logger_config.json)')
    args = parser.parse_args()

    # read configurations, hyperparameters for training and logging
    config = read_json(args.config)
    log_config = read_json(args.log_config)

    # set directories where trained model and log will be saved.
    outdir = Path(os.path.join(config['outdir'], config['name']))
    if not args.run_dir:
        run_dir = datetime.now().strftime(r'%m%d_%H%M%S')
    else:
        run_dir = args.run_dir
    checkpoint_dir = os.path.join(outdir, "checkpoint", run_dir)
    log_dir = os.path.join(outdir, "log", run_dir)

    run_setup = SetupRun(config=config,
                         checkpoint_dir=checkpoint_dir)

    log_setup = SetupLogger(config=log_config,
                            log_dir=log_dir)

    cfg_trainer = run_setup['trainer']['args']
    vizualizer_setup = TensorboardWriter(log_dir, log_setup, cfg_trainer['tensorboard'])

    # run training process
    run_training(run_setup, log_setup, vizualizer_setup=vizualizer_setup, device=config['device'])
