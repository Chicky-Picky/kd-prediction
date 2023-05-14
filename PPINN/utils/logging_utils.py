from typing import Dict
import logging
import functools
import logging.config
import os
import numpy as np


class SetupLogger:
    def __init__(self,
                 config: Dict,
                 log_dir: str = "."):
        os.makedirs(log_dir, exist_ok=True)
        config["handlers"]["info_file_handler"]["filename"] = os.path.join(log_dir, "log.log")

        self.config = config
        self.log_dir = log_dir

        logging.config.dictConfig(config)

    @staticmethod
    def get_logger(name: str,
                   log_level: int = logging.DEBUG):
        logger = logging.getLogger(name)
        logger.setLevel(log_level)
        return logger


def log_resume_checkpoint_before(logger, *args, **kwargs):
    logger.info("Loading checkpoint: {} ...".format(args[0]))


def log_resume_checkpoint_after(result, logger, *args, **kwargs):
    logger.info("Checkpoint loaded. Resume training from epoch {}".format(args[0].start_epoch))


def log_valid_epoch(result, logger, *args, **kwargs):
    log = {'epoch': args[-1]}
    log.update(result)

    for key, value in log.items():
        logger.info('    {:15s}: {}'.format(f"val_{key}", value))


def log_train_epoch(result, logger, *args, **kwargs):
    log = {'epoch': args[-1]}
    log.update(result)

    for key, value in log.items():
        logger.info('    {:15s}: {}'.format(str(key), value))


def log_train_batch(result, logger, *args, **kwargs):
    def _progress(current):
        total = len(args[0].data_loader)
        base = '[{}/{} ({:.0f}%)]'
        return base.format(current, total, 100.0 * current / total)

    _, epoch, batch_idx, data, _ = args

    log_step = int(np.sqrt(args[0].data_loader.batch_size))
    if batch_idx % log_step == 0:
        logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
            epoch,
            _progress(batch_idx),
            result.item()))


def logged(_func=None, *, logger, message_before=None, message_after=None):
    def decorator_log(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if message_before:
                message_before(logger, *args, **kwargs)
            result = func(*args, **kwargs)
            if message_after:
                message_after(result, logger, *args, **kwargs)
            return result

        return wrapper

    if _func is None:
        return decorator_log
    else:
        return decorator_log(_func)
