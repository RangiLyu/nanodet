import pytest
import torch

from nanodet.model.fpn.tan import TAN


def test_tan():
    """Tests TAN."""
    s = 64
    in_channels = [8, 16, 32]
    feat_sizes = [s // 2**i for i in range(3)]  # [64, 32, 16]
    out_channels = 8

    with pytest.raises(AssertionError):
        TAN(
            in_channels=[8, 16, 32, 64],
            out_channels=[8, 16, 32, 64],
            feature_hw=[32, 32],
            num_heads=4,
            num_encoders=1,
            mlp_ratio=2,
            dropout_ratio=0.9,
        )

    pan_model = TAN(
        in_channels=in_channels,
        out_channels=out_channels,
        feature_hw=[32, 32],
        num_heads=4,
        num_encoders=1,
        mlp_ratio=2,
        dropout_ratio=0.9,
    )

    # TAN expects a multiple levels of features per image
    feats = [
        torch.rand(1, in_channels[i], feat_sizes[i], feat_sizes[i])
        for i in range(len(in_channels))
    ]
    outs = pan_model(feats)
    assert len(outs) == 3
    for i in range(3):
        assert outs[i].shape[1] == out_channels
        assert outs[i].shape[2] == outs[i].shape[3] == s // (2**i)
