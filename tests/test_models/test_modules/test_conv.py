import pytest
import torch
import torch.nn as nn

from nanodet.model.module.conv import ConvModule


def test_conv_module():
    with pytest.raises(AssertionError):
        # conv_cfg must be a dict or None
        conv_cfg = "conv"
        ConvModule(3, 5, 2, conv_cfg=conv_cfg)

    with pytest.raises(AssertionError):
        # norm_cfg must be a dict or None
        norm_cfg = "norm"
        ConvModule(3, 5, 2, norm_cfg=norm_cfg)

    with pytest.raises(AssertionError):
        # softmax is not supported
        activation = "softmax"
        ConvModule(3, 5, 2, activation=activation)

    with pytest.raises(AssertionError):
        # softmax is not supported
        activation = dict(type="softmax")
        ConvModule(3, 5, 2, activation=activation)

    # conv + norm + act
    conv = ConvModule(3, 5, 2, norm_cfg=dict(type="BN"))
    assert hasattr(conv, "act")
    assert conv.with_norm
    assert hasattr(conv, "norm")
    x = torch.rand(1, 3, 16, 16)
    output = conv(x)
    assert output.shape == (1, 5, 15, 15)

    # conv + act
    conv = ConvModule(3, 5, 2)
    assert hasattr(conv, "act")
    assert not conv.with_norm
    assert conv.norm is None
    x = torch.rand(1, 3, 16, 16)
    output = conv(x)
    assert output.shape == (1, 5, 15, 15)

    # conv
    conv = ConvModule(3, 5, 2, activation=None)
    assert not conv.with_norm
    assert conv.norm is None
    assert not hasattr(conv, "act")
    x = torch.rand(1, 3, 16, 16)
    output = conv(x)
    assert output.shape == (1, 5, 15, 15)

    # leaky relu
    conv = ConvModule(3, 5, 3, padding=1, activation="LeakyReLU")
    assert isinstance(conv.act, nn.LeakyReLU)
    output = conv(x)
    assert output.shape == (1, 5, 16, 16)

    # PReLU
    conv = ConvModule(3, 5, 3, padding=1, activation="PReLU")
    assert isinstance(conv.act, nn.PReLU)
    output = conv(x)
    assert output.shape == (1, 5, 16, 16)


def test_bias():
    # bias: auto, without norm
    conv = ConvModule(3, 5, 2)
    assert conv.conv.bias is not None

    # bias: auto, with norm
    conv = ConvModule(3, 5, 2, norm_cfg=dict(type="BN"))
    assert conv.conv.bias is None

    # bias: False, without norm
    conv = ConvModule(3, 5, 2, bias=False)
    assert conv.conv.bias is None

    # bias: True, with norm
    with pytest.warns(UserWarning) as record:
        ConvModule(3, 5, 2, bias=True, norm_cfg=dict(type="BN"))
    assert len(record) == 1
    assert record[0].message.args[0] == "ConvModule has norm and bias at the same time"
