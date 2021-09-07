import pytest
import torch

from nanodet.model.backbone import MobileNetV2, build_backbone


def test_mobilenetv2():
    with pytest.raises(AssertionError):
        cfg = dict(name="MobileNetV2", width_mult=1.0, out_stages=(8, 9))
        build_backbone(cfg)

    input = torch.rand(1, 3, 64, 64)
    out_stages = (0, 1, 2, 3, 4, 5, 6)
    model = MobileNetV2(width_mult=1.0, out_stages=out_stages, activation="ReLU6")
    output = model(input)

    assert output[0].shape == (1, 16, 32, 32)
    assert output[1].shape == (1, 24, 16, 16)
    assert output[2].shape == (1, 32, 8, 8)
    assert output[3].shape == (1, 64, 4, 4)
    assert output[4].shape == (1, 96, 4, 4)
    assert output[5].shape == (1, 160, 2, 2)
    assert output[6].shape == (1, 1280, 2, 2)

    model = MobileNetV2(width_mult=0.75, out_stages=out_stages, activation="LeakyReLU")
    output = model(input)

    assert output[0].shape == (1, 12, 32, 32)
    assert output[1].shape == (1, 18, 16, 16)
    assert output[2].shape == (1, 24, 8, 8)
    assert output[3].shape == (1, 48, 4, 4)
    assert output[4].shape == (1, 72, 4, 4)
    assert output[5].shape == (1, 120, 2, 2)
    assert output[6].shape == (1, 1280, 2, 2)
