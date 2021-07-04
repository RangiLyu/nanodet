import pytest
import torch

from nanodet.model.backbone import RepVGG, build_backbone
from nanodet.model.backbone.repvgg import repvgg_model_convert


def test_repvgg():
    with pytest.raises(AssertionError):
        cfg = dict(name="RepVGG", arch="A3")
        build_backbone(cfg)

    with pytest.raises(AssertionError):
        RepVGG(arch="A0", out_stages=(4, 5, 6))

    input = torch.rand(1, 3, 64, 64)

    model = RepVGG(arch="A0", out_stages=(1, 2, 3, 4), activation="PReLU")
    output = model(input)

    assert output[0].shape == (1, 48, 16, 16)
    assert output[1].shape == (1, 96, 8, 8)
    assert output[2].shape == (1, 192, 4, 4)
    assert output[3].shape == (1, 1280, 2, 2)

    # test last channel
    model = RepVGG(arch="A1", out_stages=(1, 2, 3, 4), last_channel=640)
    output = model(input)

    assert output[0].shape == (1, 64, 16, 16)
    assert output[1].shape == (1, 128, 8, 8)
    assert output[2].shape == (1, 256, 4, 4)
    assert output[3].shape == (1, 640, 2, 2)

    deploy_model = RepVGG(arch="A0", deploy=True)
    deploy_model = repvgg_model_convert(model, deploy_model, save_path=None)
    dep_output = deploy_model(input)
    assert dep_output[0].shape == (1, 64, 16, 16)
    assert dep_output[1].shape == (1, 128, 8, 8)
    assert dep_output[2].shape == (1, 256, 4, 4)
    assert dep_output[3].shape == (1, 640, 2, 2)
