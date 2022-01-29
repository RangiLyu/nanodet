import torch

from nanodet.model.backbone import build_backbone
from nanodet.model.backbone.timm_wrapper import TIMMWrapper


def test_timm_wrapper():
    cfg = dict(
        name="TIMMWrapper",
        model_name="resnet18",
        features_only=True,
        pretrained=False,
        output_stride=32,
        out_indices=(1, 2, 3, 4),
    )
    model = build_backbone(cfg)

    input = torch.rand(1, 3, 64, 64)
    output = model(input)
    assert len(output) == 4
    assert output[0].shape == (1, 64, 16, 16)
    assert output[1].shape == (1, 128, 8, 8)
    assert output[2].shape == (1, 256, 4, 4)
    assert output[3].shape == (1, 512, 2, 2)

    model = TIMMWrapper(
        model_name="mobilenetv3_large_100",
        features_only=True,
        pretrained=False,
        output_stride=32,
        out_indices=(1, 2, 3, 4),
    )
    output = model(input)

    assert len(output) == 4
    assert output[0].shape == (1, 24, 16, 16)
    assert output[1].shape == (1, 40, 8, 8)
    assert output[2].shape == (1, 112, 4, 4)
    assert output[3].shape == (1, 960, 2, 2)
