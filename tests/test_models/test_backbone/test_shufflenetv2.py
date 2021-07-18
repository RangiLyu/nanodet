import pytest
import torch

from nanodet.model.backbone import ShuffleNetV2, build_backbone


def test_shufflenetv2():

    with pytest.raises(NotImplementedError):
        cfg = dict(name="ShuffleNetV2", model_size="3.0x", pretrain=False)
        build_backbone(cfg)

    with pytest.raises(AssertionError):
        ShuffleNetV2("1.0x", out_stages=(1, 2, 3))

    input = torch.rand(1, 3, 64, 64)
    model = ShuffleNetV2(model_size="0.5x", out_stages=(2, 3, 4), pretrain=True)
    output = model(input)
    assert output[0].shape == (1, 48, 8, 8)
    assert output[1].shape == (1, 96, 4, 4)
    assert output[2].shape == (1, 192, 2, 2)

    model = ShuffleNetV2(model_size="0.5x", out_stages=(3, 4), pretrain=True)
    output = model(input)
    assert output[0].shape == (1, 96, 4, 4)
    assert output[1].shape == (1, 192, 2, 2)

    model = ShuffleNetV2(model_size="1.0x", pretrain=False, with_last_conv=True)
    assert hasattr(model.stage4, "conv5")
    output = model(input)
    assert output[0].shape == (1, 116, 8, 8)
    assert output[1].shape == (1, 232, 4, 4)
    assert output[2].shape == (1, 1024, 2, 2)

    model = ShuffleNetV2(
        model_size="1.5x", pretrain=False, with_last_conv=False, activation="ReLU6"
    )
    assert not hasattr(model.stage4, "conv5")
    output = model(input)
    assert output[0].shape == (1, 176, 8, 8)
    assert output[1].shape == (1, 352, 4, 4)
    assert output[2].shape == (1, 704, 2, 2)

    model = ShuffleNetV2(model_size="2.0x", pretrain=False, with_last_conv=False)
    output = model(input)
    assert output[0].shape == (1, 244, 8, 8)
    assert output[1].shape == (1, 488, 4, 4)
    assert output[2].shape == (1, 976, 2, 2)
