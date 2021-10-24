# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os

import numpy as np
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.cloud_io import get_filesystem
from termcolor import colored

from .path import mkdir


class Logger:
    def __init__(self, local_rank, save_dir="./", use_tensorboard=True):
        mkdir(local_rank, save_dir)
        self.rank = local_rank
        fmt = (
            colored("[%(name)s]", "magenta", attrs=["bold"])
            + colored("[%(asctime)s]", "blue")
            + colored("%(levelname)s:", "green")
            + colored("%(message)s", "white")
        )
        logging.basicConfig(
            level=logging.INFO,
            filename=os.path.join(save_dir, "logs.txt"),
            filemode="w",
        )
        self.log_dir = os.path.join(save_dir, "logs")
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt, datefmt="%m-%d %H:%M:%S")
        console.setFormatter(formatter)
        logging.getLogger().addHandler(console)
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    "the dependencies to use torch.utils.tensorboard "
                    "(applicable to PyTorch 1.1 or higher)"
                ) from None
            if self.rank < 1:
                logging.info(
                    "Using Tensorboard, logs will be saved in {}".format(self.log_dir)
                )
                self.writer = SummaryWriter(log_dir=self.log_dir)

    def log(self, string):
        if self.rank < 1:
            logging.info(string)

    def scalar_summary(self, tag, phase, value, step):
        if self.rank < 1:
            self.writer.add_scalars(tag, {phase: value}, step)


class MovingAverage(object):
    def __init__(self, val, window_size=50):
        self.window_size = window_size
        self.reset()
        self.push(val)

    def reset(self):
        self.queue = []

    def push(self, val):
        self.queue.append(val)
        if len(self.queue) > self.window_size:
            self.queue.pop(0)

    def avg(self):
        return np.mean(self.queue)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, val):
        self.reset()
        self.update(val)

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


class NanoDetLightningLogger(LightningLoggerBase):
    def __init__(self, save_dir="./", **kwargs):
        super().__init__()
        self.log_dir = os.path.join(save_dir, "logs")

        self._fs = get_filesystem(save_dir)
        self._fs.makedirs(self.log_dir, exist_ok=True)
        self._init_logger()

        self._experiment = None
        self._kwargs = kwargs

    @property
    def name(self):
        return "NanoDet"

    @property
    @rank_zero_experiment
    def experiment(self):
        r"""
        Actual tensorboard object. To use TensorBoard features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.

        Example::

            self.logger.experiment.some_tensorboard_function()

        """
        if self._experiment is not None:
            return self._experiment

        assert rank_zero_only.rank == 0, "tried to init log dirs in non global_rank=0"

        # if self.root_dir:
        #     self._fs.makedirs(self.root_dir, exist_ok=True)

        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            raise ImportError(
                'Please run "pip install future tensorboard" to install '
                "the dependencies to use torch.utils.tensorboard "
                "(applicable to PyTorch 1.1 or higher)"
            )

        self._experiment = SummaryWriter(log_dir=self.log_dir, **self._kwargs)
        return self._experiment

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @rank_zero_only
    def _init_logger(self):
        self.logger = logging.getLogger(name=self.name)
        self.logger.setLevel(logging.INFO)

        # create file handler
        fh = logging.FileHandler(os.path.join(self.log_dir, "logs.txt"))
        fh.setLevel(logging.INFO)
        # set file formatter
        f_fmt = "[%(name)s] [%(asctime)s] %(levelname)s: %(message)s"
        file_formatter = logging.Formatter(f_fmt, datefmt="%m-%d %H:%M:%S")
        fh.setFormatter(file_formatter)

        # create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # set console formatter
        c_fmt = (
            colored("[%(name)s]", "magenta", attrs=["bold"])
            + colored("[%(asctime)s]", "blue")
            + colored("%(levelname)s:", "green")
            + colored("%(message)s", "white")
        )
        console_formatter = logging.Formatter(c_fmt, datefmt="%m-%d %H:%M:%S")
        ch.setFormatter(console_formatter)

        # add the handlers to the logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    @rank_zero_only
    def info(self, string):
        self.logger.info(string)

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        pass

    @rank_zero_only
    def save(self):
        super().save()

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass
