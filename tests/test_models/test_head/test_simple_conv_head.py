import torch

from nanodet.model.head import build_head
from nanodet.util.yacs import CfgNode


def test_simple_conv_head_forward():
    head_cfg = dict(
        name="SimpleConvHead",
        num_classes=80,
        input_channel=1,
        feat_channels=96,
        stacked_convs=2,
        conv_type="DWConv",
        reg_max=8,
        strides=[8, 16, 32],
    )
    cfg = CfgNode(head_cfg)
    head = build_head(cfg)
    feat = [torch.rand(1, 1, 320 // stride, 320 // stride) for stride in [8, 16, 32]]
    out = head.forward(feat)
    num_points = sum([(320 // stride) ** 2 for stride in [8, 16, 32]])
    assert out.shape == (1, num_points, 80 + (8 + 1) * 4)
