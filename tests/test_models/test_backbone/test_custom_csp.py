import pytest
import torch

from nanodet.model.backbone import CustomCspNet, build_backbone


def test_custom_csp():
    with pytest.raises(AssertionError):
        cfg = dict(
            name="CustomCspNet", net_cfg=[["Conv", 3, 32, 3, 2]], out_stages=(8, 9)
        )
        build_backbone(cfg)

    with pytest.raises(AssertionError):
        CustomCspNet(net_cfg=dict(a=1), out_stages=(0, 1), activation="ReLU6")

    input = torch.rand(1, 3, 64, 64)
    out_stages = (0, 1, 2, 3, 4, 5)
    net_cfg = [
        ["Conv", 3, 32, 3, 2],  # 1/2
        ["MaxPool", 3, 2],  # 1/4
        ["CspBlock", 32, 1, 3, 1],  # 1/4
        ["CspBlock", 64, 2, 3, 2],  # 1/8
        ["CspBlock", 128, 2, 3, 2],  # 1/16
        ["CspBlock", 256, 3, 3, 2],  # 1/32
    ]
    model = CustomCspNet(net_cfg=net_cfg, out_stages=out_stages, activation="ReLU6")
    output = model(input)

    assert output[0].shape == (1, 32, 32, 32)
    assert output[1].shape == (1, 32, 16, 16)
    assert output[2].shape == (1, 64, 16, 16)
    assert output[3].shape == (1, 128, 8, 8)
    assert output[4].shape == (1, 256, 4, 4)
    assert output[5].shape == (1, 512, 2, 2)
