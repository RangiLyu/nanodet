import numpy as np
import torch

from nanodet.model.head import build_head
from nanodet.util.yacs import CfgNode


def test_nanodet_plus_head_loss():
    head_cfg = dict(
        name="NanoDetPlusHead",
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
        name="NanoDetPlusHead",
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

    # Test that empty ground truth encourages the network to predict background
    meta = dict(
        img=torch.rand((1, 3, 320, 320)),
        gt_bboxes=[np.random.random((0, 4))],
        gt_labels=[np.array([])],
    )
    loss, empty_gt_losses = head.loss(preds, meta)
    # When there is no truth, the cls loss should be nonzero but there should
    # be no box loss.
    empty_qfl_loss = empty_gt_losses["loss_qfl"]
    empty_box_loss = empty_gt_losses["loss_bbox"]
    empty_dfl_loss = empty_gt_losses["loss_dfl"]
    assert empty_qfl_loss.item() > 0
    assert (
        empty_box_loss.item() == 0
    ), "there should be no box loss when there are no true boxes"
    assert (
        empty_dfl_loss.item() == 0
    ), "there should be no dfl loss when there are no true boxes"

    # When truth is non-empty then both cls and box loss should be nonzero for
    # random inputs
    gt_bboxes = [
        np.array([[23.6667, 23.8757, 238.6326, 151.8874]], dtype=np.float32),
    ]
    gt_labels = [np.array([2])]
    meta = dict(
        img=torch.rand((1, 3, 320, 320)), gt_bboxes=gt_bboxes, gt_labels=gt_labels
    )
    loss, one_gt_losses = head.loss(preds, meta)
    onegt_qfl_loss = one_gt_losses["loss_qfl"]
    onegt_box_loss = one_gt_losses["loss_bbox"]
    onegt_dfl_loss = one_gt_losses["loss_dfl"]
    assert onegt_qfl_loss.item() > 0, "qfl loss should be non-zero"
    assert onegt_box_loss.item() > 0, "box loss should be non-zero"
    assert onegt_dfl_loss.item() > 0, "dfl loss should be non-zero"

    # test aux input
    gt_bboxes = [
        np.array([[23.6667, 23.8757, 238.6326, 151.8874]], dtype=np.float32),
    ]
    gt_labels = [np.array([2])]
    meta = dict(
        img=torch.rand((1, 3, 320, 320)), gt_bboxes=gt_bboxes, gt_labels=gt_labels
    )
    loss, one_gt_losses = head.loss(preds, meta, aux_preds=preds)
    onegt_qfl_loss = one_gt_losses["loss_qfl"]
    onegt_box_loss = one_gt_losses["loss_bbox"]
    onegt_dfl_loss = one_gt_losses["loss_dfl"]
    onegt_aux_qfl_loss = one_gt_losses["aux_loss_qfl"]
    onegt_aux_box_loss = one_gt_losses["aux_loss_bbox"]
    onegt_aux_dfl_loss = one_gt_losses["aux_loss_dfl"]
    assert onegt_qfl_loss.item() > 0, "qfl loss should be non-zero"
    assert onegt_box_loss.item() > 0, "box loss should be non-zero"
    assert onegt_dfl_loss.item() > 0, "dfl loss should be non-zero"
    assert onegt_aux_qfl_loss.item() > 0, "aux_qfl loss should be non-zero"
    assert onegt_aux_box_loss.item() > 0, "aux_box loss should be non-zero"
    assert onegt_aux_dfl_loss.item() > 0, "aux_dfl loss should be non-zero"
