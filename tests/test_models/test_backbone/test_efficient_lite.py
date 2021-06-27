import pytest
import torch

from nanodet.model.backbone import EfficientNetLite, build_backbone


def test_efficientnet_lite():
    with pytest.raises(AssertionError):
        cfg = dict(
            name="EfficientNetLite",
            model_name="efficientnet_lite0",
            out_stages=(7, 8, 9),
        )
        build_backbone(cfg)

    with pytest.raises(AssertionError):
        EfficientNetLite(model_name="efficientnet_lite9")

    input = torch.rand(1, 3, 64, 64)

    model = EfficientNetLite(
        model_name="efficientnet_lite0", out_stages=(0, 1, 2, 3, 4, 5, 6), pretrain=True
    )
    output = model(input)
    assert output[0].shape == (1, 16, 32, 32)
    assert output[1].shape == (1, 24, 16, 16)
    assert output[2].shape == (1, 40, 8, 8)
    assert output[3].shape == (1, 80, 4, 4)
    assert output[4].shape == (1, 112, 4, 4)
    assert output[5].shape == (1, 192, 2, 2)
    assert output[6].shape == (1, 320, 2, 2)

    model = EfficientNetLite(
        model_name="efficientnet_lite1",
        out_stages=(0, 1, 2, 3, 4, 5, 6),
        pretrain=False,
    )
    output = model(input)
    assert output[0].shape == (1, 16, 32, 32)
    assert output[1].shape == (1, 24, 16, 16)
    assert output[2].shape == (1, 40, 8, 8)
    assert output[3].shape == (1, 80, 4, 4)
    assert output[4].shape == (1, 112, 4, 4)
    assert output[5].shape == (1, 192, 2, 2)
    assert output[6].shape == (1, 320, 2, 2)

    model = EfficientNetLite(
        model_name="efficientnet_lite2",
        out_stages=(0, 1, 2, 3, 4, 5, 6),
        activation="ReLU",
        pretrain=False,
    )
    output = model(input)
    assert output[0].shape == (1, 16, 32, 32)
    assert output[1].shape == (1, 24, 16, 16)
    assert output[2].shape == (1, 48, 8, 8)
    assert output[3].shape == (1, 88, 4, 4)
    assert output[4].shape == (1, 120, 4, 4)
    assert output[5].shape == (1, 208, 2, 2)
    assert output[6].shape == (1, 352, 2, 2)

    model = EfficientNetLite(
        model_name="efficientnet_lite3",
        out_stages=(0, 1, 2, 3, 4, 5, 6),
        pretrain=False,
    )
    output = model(input)
    assert output[0].shape == (1, 24, 32, 32)
    assert output[1].shape == (1, 32, 16, 16)
    assert output[2].shape == (1, 48, 8, 8)
    assert output[3].shape == (1, 96, 4, 4)
    assert output[4].shape == (1, 136, 4, 4)
    assert output[5].shape == (1, 232, 2, 2)
    assert output[6].shape == (1, 384, 2, 2)

    model = EfficientNetLite(
        model_name="efficientnet_lite4",
        out_stages=(0, 1, 2, 3, 4, 5, 6),
        pretrain=False,
    )
    output = model(input)
    assert output[0].shape == (1, 24, 32, 32)
    assert output[1].shape == (1, 32, 16, 16)
    assert output[2].shape == (1, 56, 8, 8)
    assert output[3].shape == (1, 112, 4, 4)
    assert output[4].shape == (1, 160, 4, 4)
    assert output[5].shape == (1, 272, 2, 2)
    assert output[6].shape == (1, 448, 2, 2)
