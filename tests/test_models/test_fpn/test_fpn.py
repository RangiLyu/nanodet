import pytest
import torch

from nanodet.model.fpn.fpn import FPN


def test_fpn():
    """Tests fpn."""
    s = 64
    in_channels = [8, 16, 32, 64]
    feat_sizes = [s // 2**i for i in range(4)]  # [64, 32, 16, 8]
    out_channels = 8
    # `num_outs` is not equal to len(in_channels) - start_level
    with pytest.raises(AssertionError):
        FPN(
            in_channels=in_channels,
            out_channels=out_channels,
            start_level=1,
            num_outs=2,
        )

    # `end_level` is larger than len(in_channels) - 1
    with pytest.raises(AssertionError):
        FPN(
            in_channels=in_channels,
            out_channels=out_channels,
            start_level=1,
            end_level=4,
            num_outs=2,
        )

    # `num_outs` is not equal to end_level - start_level
    with pytest.raises(AssertionError):
        FPN(
            in_channels=in_channels,
            out_channels=out_channels,
            start_level=1,
            end_level=3,
            num_outs=1,
        )

    fpn_model = FPN(
        in_channels=in_channels, out_channels=out_channels, start_level=1, num_outs=3
    )

    # FPN expects a multiple levels of features per image
    feats = [
        torch.rand(1, in_channels[i], feat_sizes[i], feat_sizes[i])
        for i in range(len(in_channels))
    ]
    outs = fpn_model(feats)
    assert len(outs) == fpn_model.num_outs
    for i in range(fpn_model.num_outs):
        assert outs[i].shape[1] == out_channels
        assert outs[i].shape[2] == outs[i].shape[3] == s // (2 ** (i + 1))
