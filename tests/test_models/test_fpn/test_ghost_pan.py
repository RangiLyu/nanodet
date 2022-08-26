import pytest
import torch

from nanodet.model.fpn.ghost_pan import GhostPAN


def test_ghost_pan():
    """Tests GhostPAN."""
    s = 64
    in_channels = [8, 16, 32, 64]
    feat_sizes = [s // 2**i for i in range(4)]  # [64, 32, 16, 8]
    out_channels = 8
    # `num_extra_level` >= 0
    with pytest.raises(AssertionError):
        GhostPAN(in_channels=in_channels, out_channels=out_channels, num_extra_level=-1)

    # `num_blocks` > 0
    with pytest.raises(AssertionError):
        GhostPAN(in_channels=in_channels, out_channels=out_channels, num_blocks=0)

    pan_model = GhostPAN(
        in_channels=in_channels,
        out_channels=out_channels,
        num_extra_level=1,
        num_blocks=1,
    )

    # PAN expects a multiple levels of features per image
    feats = [
        torch.rand(1, in_channels[i], feat_sizes[i], feat_sizes[i])
        for i in range(len(in_channels))
    ]
    outs = pan_model(feats)
    assert len(outs) == len(in_channels) + 1
    for i in range(len(in_channels)):
        assert outs[i].shape[1] == out_channels
        assert outs[i].shape[2] == outs[i].shape[3] == s // (2**i)
    assert outs[-1].shape[1] == out_channels
    assert outs[-1].shape[2] == outs[-1].shape[3] == s // 2 ** len(in_channels)

    # test non-default values
    pan_model = GhostPAN(
        in_channels=in_channels,
        out_channels=out_channels,
        num_extra_level=1,
        num_blocks=1,
        use_depthwise=True,
        expand=2,
        use_res=True,
    )
    # PAN expects a multiple levels of features per image
    assert (
        pan_model.downsamples[0].depthwise.groups
        == pan_model.downsamples[0].depthwise.in_channels
    )
    feats = [
        torch.rand(1, in_channels[i], feat_sizes[i], feat_sizes[i])
        for i in range(len(in_channels))
    ]
    outs = pan_model(feats)
    assert len(outs) == len(in_channels) + 1
    for i in range(len(in_channels)):
        assert outs[i].shape[1] == out_channels
        assert outs[i].shape[2] == outs[i].shape[3] == s // (2**i)
    assert outs[-1].shape[1] == out_channels
    assert outs[-1].shape[2] == outs[-1].shape[3] == s // 2 ** len(in_channels)
