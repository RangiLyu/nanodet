import pytest
import torch
import torch.nn as nn

from nanodet.model.module.init_weights import (
    constant_init,
    kaiming_init,
    normal_init,
    xavier_init,
)


def test_constant_init():
    conv_module = nn.Conv2d(3, 16, 3)
    constant_init(conv_module, 0.1)
    assert conv_module.weight.allclose(torch.full_like(conv_module.weight, 0.1))
    assert conv_module.bias.allclose(torch.zeros_like(conv_module.bias))
    conv_module_no_bias = nn.Conv2d(3, 16, 3, bias=False)
    constant_init(conv_module_no_bias, 0.1)
    assert conv_module.weight.allclose(torch.full_like(conv_module.weight, 0.1))


def test_xavier_init():
    conv_module = nn.Conv2d(3, 16, 3)
    xavier_init(conv_module, bias=0.1)
    assert conv_module.bias.allclose(torch.full_like(conv_module.bias, 0.1))
    xavier_init(conv_module, distribution="uniform")
    with pytest.raises(AssertionError):
        xavier_init(conv_module, distribution="student-t")
    conv_module_no_bias = nn.Conv2d(3, 16, 3, bias=False)
    xavier_init(conv_module_no_bias)


def test_normal_init():
    conv_module = nn.Conv2d(3, 16, 3)
    normal_init(conv_module, bias=0.1)
    assert conv_module.bias.allclose(torch.full_like(conv_module.bias, 0.1))
    conv_module_no_bias = nn.Conv2d(3, 16, 3, bias=False)
    normal_init(conv_module_no_bias)


def test_kaiming_init():
    conv_module = nn.Conv2d(3, 16, 3)
    kaiming_init(conv_module, bias=0.1)
    assert conv_module.bias.allclose(torch.full_like(conv_module.bias, 0.1))
    kaiming_init(conv_module, distribution="uniform")
    with pytest.raises(AssertionError):
        kaiming_init(conv_module, distribution="student-t")
    conv_module_no_bias = nn.Conv2d(3, 16, 3, bias=False)
    kaiming_init(conv_module_no_bias)
