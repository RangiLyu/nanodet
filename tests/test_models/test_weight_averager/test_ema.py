# Copyright 2021 RangiLyu. All rights reserved.
import copy

import pytest
import torch
import torch.nn as nn

from nanodet.model.weight_averager import ExpMovingAverager


def test_ema():
    # test invalid input
    with pytest.raises(ValueError):
        ExpMovingAverager(-1)
    # test invalid input
    with pytest.raises(ValueError):
        ExpMovingAverager(10)

    class DummyModel(nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.fc = nn.Linear(10, 10)

        def forward(self, x):
            return self.fc(x)

    averager = ExpMovingAverager(0.5)
    model = DummyModel()
    ema_model = copy.deepcopy(model)

    averager.load_from(model)
    assert averager.has_inited()

    averager.update(model, 1)
    averager.apply_to(ema_model)
    assert torch.allclose(ema_model.fc.weight, model.fc.weight)
