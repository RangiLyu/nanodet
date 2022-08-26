import pytest
import torch

from nanodet.model.fpn.pan import PAN


def test_pan():
    """Tests PAN."""
    s = 64
    in_channels = [8, 16, 32, 64]
    feat_sizes = [s // 2**i for i in range(4)]  # [64, 32, 16, 8]
    out_channels = 8
    # `num_outs` is not equal to len(in_channels) - start_level
    with pytest.raises(AssertionError):
        PAN(
            in_channels=in_channels,
            out_channels=out_channels,
            start_level=1,
            num_outs=2,
        )

    # `end_level` is larger than len(in_channels) - 1
    with pytest.raises(AssertionError):
        PAN(
            in_channels=in_channels,
            out_channels=out_channels,
            start_level=1,
            end_level=4,
            num_outs=2,
        )

    # `num_outs` is not equal to end_level - start_level
    with pytest.raises(AssertionError):
        PAN(
            in_channels=in_channels,
            out_channels=out_channels,
            start_level=1,
            end_level=3,
            num_outs=1,
        )

    pan_model = PAN(
        in_channels=in_channels, out_channels=out_channels, start_level=1, num_outs=3
    )

    # PAN expects a multiple levels of features per image
    feats = [
        torch.rand(1, in_channels[i], feat_sizes[i], feat_sizes[i])
        for i in range(len(in_channels))
    ]
    outs = pan_model(feats)
    assert len(outs) == pan_model.num_outs
    for i in range(pan_model.num_outs):
        assert outs[i].shape[1] == out_channels
        assert outs[i].shape[2] == outs[i].shape[3] == s // (2 ** (i + 1))
