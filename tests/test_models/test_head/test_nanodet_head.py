import torch

from nanodet.model.head import build_head
from nanodet.util.yacs import CfgNode


def test_gfl_head_loss():
    head_cfg = dict(
        name="NanoDetHead",
        num_classes=80,
        input_channel=1,
        feat_channels=96,
        stacked_convs=2,
        conv_type="DWConv",
        reg_max=8,
        strides=[8, 16, 32],
        loss=dict(
            loss_qfl=dict(
                name="QualityFocalLoss", use_sigmoid=True, beta=2.0, loss_weight=1.0
            ),
            loss_dfl=dict(name="DistributionFocalLoss", loss_weight=0.25),
            loss_bbox=dict(name="GIoULoss", loss_weight=2.0),
        ),
    )
    cfg = CfgNode(head_cfg)

    head = build_head(cfg)
    feat = [torch.rand(1, 1, 320 // stride, 320 // stride) for stride in [8, 16, 32]]

    preds = head.forward(feat)
    num_points = sum([(320 // stride) ** 2 for stride in [8, 16, 32]])
    assert preds.shape == (1, num_points, 80 + (8 + 1) * 4)

    head_cfg = dict(
        name="NanoDetHead",
        num_classes=20,
        input_channel=1,
        feat_channels=96,
        stacked_convs=2,
        conv_type="Conv",
        reg_max=5,
        share_cls_reg=False,
        strides=[8, 16, 32],
        loss=dict(
            loss_qfl=dict(
                name="QualityFocalLoss", use_sigmoid=True, beta=2.0, loss_weight=1.0
            ),
            loss_dfl=dict(name="DistributionFocalLoss", loss_weight=0.25),
            loss_bbox=dict(name="GIoULoss", loss_weight=2.0),
        ),
    )
    cfg = CfgNode(head_cfg)
    head = build_head(cfg)

    preds = head.forward(feat)
    num_points = sum([(320 // stride) ** 2 for stride in [8, 16, 32]])
    assert preds.shape == (1, num_points, 20 + (5 + 1) * 4)
