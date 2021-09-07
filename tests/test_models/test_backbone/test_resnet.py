import pytest
import torch

from nanodet.model.backbone import ResNet, build_backbone


def test_resnet():
    with pytest.raises(KeyError):
        cfg = dict(name="ResNet", depth=15)
        build_backbone(cfg)

    with pytest.raises(AssertionError):
        ResNet(depth=18, out_stages=(4, 5, 6))

    input = torch.rand(1, 3, 64, 64)

    model = ResNet(depth=18, out_stages=(1, 2, 3, 4), activation="PReLU", pretrain=True)
    output = model(input)

    assert output[0].shape == (1, 64, 16, 16)
    assert output[1].shape == (1, 128, 8, 8)
    assert output[2].shape == (1, 256, 4, 4)
    assert output[3].shape == (1, 512, 2, 2)

    model = ResNet(
        depth=34, out_stages=(1, 2, 3, 4), activation="LeakyReLU", pretrain=False
    )
    output = model(input)
    assert output[0].shape == (1, 64, 16, 16)
    assert output[1].shape == (1, 128, 8, 8)
    assert output[2].shape == (1, 256, 4, 4)
    assert output[3].shape == (1, 512, 2, 2)

    model = ResNet(depth=50, out_stages=(1, 2, 3, 4), pretrain=False)
    output = model(input)
    assert output[0].shape == (1, 256, 16, 16)
    assert output[1].shape == (1, 512, 8, 8)
    assert output[2].shape == (1, 1024, 4, 4)
    assert output[3].shape == (1, 2048, 2, 2)

    model = ResNet(depth=101, out_stages=(1, 2, 3, 4), pretrain=False)
    output = model(input)
    assert output[0].shape == (1, 256, 16, 16)
    assert output[1].shape == (1, 512, 8, 8)
    assert output[2].shape == (1, 1024, 4, 4)
    assert output[3].shape == (1, 2048, 2, 2)
