import numpy as np
import pytest
import torch

from nanodet.model.module.conv import RepVGGConvModule


def test_repvgg_conv():
    # test activation type
    with pytest.raises(AssertionError):
        activation = dict(type="softmax")
        RepVGGConvModule(3, 5, 3, padding=1, activation=activation)

    # repvgg only support 3x3 conv
    with pytest.raises(AssertionError):
        RepVGGConvModule(3, 5, 2, activation="ReLU")

    with pytest.raises(AssertionError):
        RepVGGConvModule(3, 5, 3, padding=0, activation="ReLU")

    training_conv = RepVGGConvModule(3, 5, 3, deploy=False)
    assert hasattr(training_conv, "rbr_identity")
    assert hasattr(training_conv, "rbr_dense")
    assert hasattr(training_conv, "rbr_1x1")
    assert not hasattr(training_conv, "rbr_reparam")

    deploy_conv = RepVGGConvModule(3, 5, 3, deploy=True)
    assert not hasattr(deploy_conv, "rbr_identity")
    assert not hasattr(deploy_conv, "rbr_dense")
    assert not hasattr(deploy_conv, "rbr_1x1")
    assert hasattr(deploy_conv, "rbr_reparam")

    converted_weights = {}
    deploy_conv.load_state_dict(training_conv.state_dict(), strict=False)
    kernel, bias = training_conv.repvgg_convert()
    converted_weights["rbr_reparam.weight"] = kernel
    converted_weights["rbr_reparam.bias"] = bias
    for name, param in deploy_conv.named_parameters():
        print("deploy param: ", name, param.size(), np.mean(converted_weights[name]))
        param.data = torch.from_numpy(converted_weights[name]).float()

    x = torch.rand(1, 3, 16, 16)
    train_out = training_conv(x)
    deploy_out = deploy_conv(x)
    assert train_out.shape == (1, 5, 16, 16)
    assert deploy_out.shape == (1, 5, 16, 16)
