import numpy as np
import torch
import torch.nn as nn

from nanodet.trainer.task import TrainingTask
from nanodet.util import cfg, load_config


class DummyModule(nn.Module):
    current_epoch = 0
    global_step = 0
    local_rank = 0
    use_ddp = False

    def save_checkpoint(self, *args, **kwargs):
        pass


class DummyRunner:
    def __init__(self, task):
        self.task = task

    def test(self):
        optimizer = self.task.configure_optimizers()

        def optimizers():
            return optimizer

        self.task.optimizers = optimizers
        # self.task.trainer = DummyModule()

        self.task.on_train_start()
        assert self.task.current_epoch == 0
        assert self.task.lr_scheduler.last_epoch == 0

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
            "gt_labels": [np.array([1]), np.array([1, 2])],
            "warp_matrix": [np.eye(3), np.eye(3)],
        }

        def func(*args, **kwargs):
            pass

        self.task.scalar_summary = func
        self.task.training_step(dummy_batch, 0)

        self.task.trainer = DummyModule()
        self.task.optimizer_step(optimizer=optimizer)
        self.task.training_epoch_end([])
        assert self.task.lr_scheduler.last_epoch == 1

        self.task.validation_step(dummy_batch, 0)
        self.task.validation_epoch_end([])

        self.task.test_step(dummy_batch, 0)
        self.task.test_epoch_end([])


def test_lightning_training_task():
    load_config(cfg, "./config/nanodet-m.yml")
    task = TrainingTask(cfg)
    runner = DummyRunner(task)
    runner.test()
