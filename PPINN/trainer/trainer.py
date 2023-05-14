from utils.logging_utils import logged, SetupLogger, log_train_epoch, log_valid_epoch, log_train_batch, \
    log_resume_checkpoint_before, log_resume_checkpoint_after
from utils.util import MetricTracker
import torch
import numpy as np

logger = SetupLogger.get_logger(__name__)


class Trainer:
    """
    Trainer class
    """

    def __init__(self,
                 model,
                 criterion,
                 metric_ftns,
                 optimizer,
                 run_setup,
                 vizualizer,
                 device,
                 data_loader,
                 valid_data_loader,
                 lr_scheduler=None
                 ):
        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.config = run_setup
        self.device = device
        self.model.to(self.device)
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler

        self.writer = vizualizer
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        cfg_trainer = run_setup['trainer']['args']
        self.epochs = cfg_trainer['epochs']
        self.checkpoint_freq = cfg_trainer['checkpoint_freq']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        self.mnt_mode, self.mnt_metric = self.monitor.split()
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = np.inf if self.mnt_mode == 'min' else -np.inf
        self.early_stop = cfg_trainer.get('early_stop', np.inf)
        if self.early_stop <= 0:
            self.early_stop = np.inf

        self.start_epoch = 1
        self.checkpoint_dir = run_setup["checkpoint_dir"]

        if run_setup.resume is not None:
            self.resume_checkpoint(run_setup.resume)

        self.lr_scheduler = lr_scheduler

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self.train_epoch(epoch)
            val_log = self.valid_epoch(epoch)
            result.update(**{'val_' + k: v for k, v in val_log.items()})

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            # check whether model performance improved or not, according to specified metric(mnt_metric)
            improved = (self.mnt_mode == 'min' and result[self.mnt_metric] <= self.mnt_best) or \
                       (self.mnt_mode == 'max' and result[self.mnt_metric] >= self.mnt_best)

            if improved:
                self.mnt_best = result[self.mnt_metric]
                not_improved_count = 0
                best = True
            else:
                not_improved_count += 1

            if not_improved_count > self.early_stop:
                logger.info("Validation performance didn\'t improve for {} epochs. "
                            "Training stops.".format(self.early_stop))
                break

            if epoch % self.checkpoint_freq == 0:
                self.save_checkpoint(epoch, save_best=best)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    @logged(logger=logger, message_after=log_train_epoch)
    def train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            self.train_batch(epoch, batch_idx, data, target)

        return self.train_metrics.result()

    @logged(logger=logger, message_after=log_valid_epoch)
    def valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                self.valid_batch(epoch, batch_idx, data, target)
        return self.valid_metrics.result()

    @logged(logger=logger, message_after=log_train_batch)
    def train_batch(self, epoch, batch_idx, data, target):
        data, target = data.to(self.device), target.to(self.device)

        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()

        self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)

        self.train_metrics.update('loss', loss.item())
        for met in self.metric_ftns:
            self.train_metrics.update(met.__name__, met(output, target))

        return loss

    def valid_batch(self, epoch, batch_idx, data, target):
        data, target = data.to(self.device), target.to(self.device)

        output = self.model(data)
        loss = self.criterion(output, target)

        self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
        self.valid_metrics.update('loss', loss.item())
        for met in self.metric_ftns:
            self.valid_metrics.update(met.__name__, met(output, target))
        return self.valid_metrics

    def save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            logger.info("Saving current best: model_best.pth ...")

    @logged(logger=logger, message_before=log_resume_checkpoint_before, message_after=log_resume_checkpoint_after)
    def resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # # load architecture params from checkpoint.
        assert checkpoint['config']['arch'] == self.config[
            'arch'], "Architecture configuration given in config file is different from that of checkpoint"

        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        assert checkpoint['config']['optimizer']['type'] == self.config['optimizer'][
            'type'], "Optimizer type given in config file is different from that of checkpoint."

        self.optimizer.load_state_dict(checkpoint['optimizer'])
