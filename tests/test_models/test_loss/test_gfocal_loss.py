import pytest
import torch

from nanodet.model.loss.gfocal_loss import DistributionFocalLoss, QualityFocalLoss


def test_qfl():
    with pytest.raises(AssertionError):
        QualityFocalLoss(use_sigmoid=False)

    label = torch.randint(low=0, high=7, size=(10,))
    score = torch.rand((10,))
    pred = torch.rand((10, 7))
    target = (label, score)
    weight = torch.zeros(10)

    loss = QualityFocalLoss()(pred, target, weight)
    assert loss == 0.0

    loss = QualityFocalLoss()(pred, target, weight, reduction_override="sum")
    assert loss == 0.0


def test_dfl():

    pred = torch.rand((10, 7))
    target = torch.rand((10,))
    weight = torch.zeros(10)

    loss = DistributionFocalLoss()(pred, target, weight)
    assert loss == 0.0

    loss = DistributionFocalLoss()(pred, target, weight, reduction_override="sum")
    assert loss == 0.0
