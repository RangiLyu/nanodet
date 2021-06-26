import pytest
import torch

from nanodet.model.loss.iou_loss import (
    BoundedIoULoss,
    CIoULoss,
    DIoULoss,
    GIoULoss,
    IoULoss,
)


@pytest.mark.parametrize(
    "loss_class", [IoULoss, BoundedIoULoss, GIoULoss, DIoULoss, CIoULoss]
)
def test_iou_type_loss_zeros_weight(loss_class):
    pred = torch.rand((10, 4))
    target = torch.rand((10, 4))
    weight = torch.zeros(10)

    with pytest.raises(AssertionError):
        loss_class()(pred, target, reduction_override="2333")

    loss = loss_class()(pred, target, weight)
    assert loss == 0.0

    weight = torch.rand(10)
    loss = loss_class()(pred, target, weight)
    assert loss != 0.0
