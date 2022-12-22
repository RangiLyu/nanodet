import tempfile
from unittest.mock import Mock

import numpy as np
import torch

from nanodet.trainer.task import TrainingTask
from nanodet.util import NanoDetLightningLogger, cfg, load_config


class DummyRunner:
    def __init__(self, task):
        self.task = task

    def test(self):
        trainer = Mock()
        trainer.current_epoch = 0
        trainer.global_step = 0
        trainer.local_rank = 0
        trainer.use_ddp = False
        trainer.loggers = [NanoDetLightningLogger(tempfile.TemporaryDirectory().name)]
        trainer.num_val_batches = [1]

        optimizer = self.task.configure_optimizers()["optimizer"]

        trainer.optimizers = [optimizer]
        self.task._trainer = trainer

        self.task.on_train_start()
        assert self.task.current_epoch == 0

        dummy_batch = {
            "img": torch.randn((2, 3, 32, 32)),
            "img_info": {
                "height": torch.randn(2),
                "width": torch.randn(2),
                "id": torch.from_numpy(np.array([0, 1])),
            },
            "gt_bboxes": [
                np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
                np.array(
                    [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]], dtype=np.float32
                ),
            ],
            "gt_bboxes_ignore": [
                np.array([[3.0, 4.0, 5.0, 6.0]], dtype=np.float32),
                np.array(
                    [[7.0, 8.0, 9.0, 10.0], [7.0, 8.0, 9.0, 10.0]], dtype=np.float32
                ),
            ],
            "gt_labels": [np.array([1]), np.array([1, 2])],
            "warp_matrix": [np.eye(3), np.eye(3)],
        }

        def func(*args, **kwargs):
            pass

        self.task.scalar_summary = func
        self.task.training_step(dummy_batch, 0)

        self.task.optimizer_step(optimizer=optimizer)
        self.task.training_epoch_end([])

        self.task.validation_step(dummy_batch, 0)
        self.task.validation_epoch_end([])

        self.task.test_step(dummy_batch, 0)
        self.task.test_epoch_end([])


def test_lightning_training_task():
    load_config(cfg, "./config/legacy_v0.x_configs/nanodet-m.yml")
    task = TrainingTask(cfg)
    runner = DummyRunner(task)
    runner.test()
