import pytest
import torch

from nanodet.model.backbone import GhostNet, build_backbone


def test_ghostnet():
    with pytest.raises(AssertionError):
        cfg = dict(name="GhostNet", width_mult=1.0, out_stages=(11, 12), pretrain=False)
        build_backbone(cfg)

    input = torch.rand(1, 3, 64, 64)
    out_stages = [i for i in range(10)]
    model = GhostNet(
        width_mult=1.0, out_stages=out_stages, activation="ReLU6", pretrain=True
    )
    output = model(input)

    assert output[0].shape == torch.Size([1, 16, 32, 32])
    assert output[1].shape == torch.Size([1, 24, 16, 16])
    assert output[2].shape == torch.Size([1, 24, 16, 16])
    assert output[3].shape == torch.Size([1, 40, 8, 8])
    assert output[4].shape == torch.Size([1, 40, 8, 8])
    assert output[5].shape == torch.Size([1, 80, 4, 4])
    assert output[6].shape == torch.Size([1, 112, 4, 4])
    assert output[7].shape == torch.Size([1, 160, 2, 2])
    assert output[8].shape == torch.Size([1, 160, 2, 2])
    assert output[9].shape == torch.Size([1, 960, 2, 2])

    model = GhostNet(
        width_mult=0.75, out_stages=out_stages, activation="LeakyReLU", pretrain=False
    )
    output = model(input)

    assert output[0].shape == torch.Size([1, 12, 32, 32])
    assert output[1].shape == torch.Size([1, 20, 16, 16])
    assert output[2].shape == torch.Size([1, 20, 16, 16])
    assert output[3].shape == torch.Size([1, 32, 8, 8])
    assert output[4].shape == torch.Size([1, 32, 8, 8])
    assert output[5].shape == torch.Size([1, 60, 4, 4])
    assert output[6].shape == torch.Size([1, 84, 4, 4])
    assert output[7].shape == torch.Size([1, 120, 2, 2])
    assert output[8].shape == torch.Size([1, 120, 2, 2])
    assert output[9].shape == torch.Size([1, 720, 2, 2])
