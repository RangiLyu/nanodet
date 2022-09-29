from unittest import TestCase

import torch
import torch.nn as nn

from nanodet.optim import build_optimizer


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 14, 3)
        self.bn = nn.BatchNorm2d(32)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class TestOptimBuilder(TestCase):
    def test_build_optimizer(self):
        model = ToyModel()
        config = dict(
            name="SGD",
            lr=0.001,
            momentum=0.9,
            param_level_cfg=dict(conv=dict(lr_mult=0.3, decay_mult=0.7)),
            no_bias_decay=True,
            no_norm_decay=True,
        )
        optimizer = build_optimizer(model, config)
        assert isinstance(optimizer, torch.optim.SGD)
