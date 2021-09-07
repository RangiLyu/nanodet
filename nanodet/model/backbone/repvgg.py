"""
@article{ding2101repvgg,
  title={RepVGG: Making VGG-style ConvNets Great Again},
  author={Ding, Xiaohan and Zhang, Xiangyu and Ma, Ningning and Han,
          Jungong and Ding, Guiguang and Sun, Jian},
  journal={arXiv preprint arXiv:2101.03697}}
RepVGG Backbone from paper RepVGG: Making VGG-style ConvNets Great Again
Code from https://github.com/DingXiaoH/RepVGG
"""

import numpy as np
import torch
import torch.nn as nn

from nanodet.model.module.conv import RepVGGConvModule

optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {layer: 2 for layer in optional_groupwise_layers}
g4_map = {layer: 4 for layer in optional_groupwise_layers}

model_param = {
    "RepVGG-A0": dict(
        num_blocks=[2, 4, 14, 1],
        width_multiplier=[0.75, 0.75, 0.75, 2.5],
        override_groups_map=None,
    ),
    "RepVGG-A1": dict(
        num_blocks=[2, 4, 14, 1],
        width_multiplier=[1, 1, 1, 2.5],
        override_groups_map=None,
    ),
    "RepVGG-A2": dict(
        num_blocks=[2, 4, 14, 1],
        width_multiplier=[1.5, 1.5, 1.5, 2.75],
        override_groups_map=None,
    ),
    "RepVGG-B0": dict(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[1, 1, 1, 2.5],
        override_groups_map=None,
    ),
    "RepVGG-B1": dict(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[2, 2, 2, 4],
        override_groups_map=None,
    ),
    "RepVGG-B1g2": dict(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[2, 2, 2, 4],
        override_groups_map=g2_map,
    ),
    "RepVGG-B1g4": dict(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[2, 2, 2, 4],
        override_groups_map=g4_map,
    ),
    "RepVGG-B2": dict(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[2.5, 2.5, 2.5, 5],
        override_groups_map=None,
    ),
    "RepVGG-B2g2": dict(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[2.5, 2.5, 2.5, 5],
        override_groups_map=g2_map,
    ),
    "RepVGG-B2g4": dict(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[2.5, 2.5, 2.5, 5],
        override_groups_map=g4_map,
    ),
    "RepVGG-B3": dict(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[3, 3, 3, 5],
        override_groups_map=None,
    ),
    "RepVGG-B3g2": dict(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[3, 3, 3, 5],
        override_groups_map=g2_map,
    ),
    "RepVGG-B3g4": dict(
        num_blocks=[4, 6, 16, 1],
        width_multiplier=[3, 3, 3, 5],
        override_groups_map=g4_map,
    ),
}


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module(
        "conv",
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
    )
    result.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGG(nn.Module):
    def __init__(
        self,
        arch,
        out_stages=(1, 2, 3, 4),
        activation="ReLU",
        deploy=False,
        last_channel=None,
    ):
        super(RepVGG, self).__init__()
        # TODO: Update code to Xiaohan's repo
        model_name = "RepVGG-" + arch
        assert model_name in model_param
        assert set(out_stages).issubset((1, 2, 3, 4))
        num_blocks = model_param[model_name]["num_blocks"]
        width_multiplier = model_param[model_name]["width_multiplier"]
        assert len(width_multiplier) == 4
        self.out_stages = out_stages
        self.activation = activation
        self.deploy = deploy
        self.override_groups_map = (
            model_param[model_name]["override_groups_map"] or dict()
        )

        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = RepVGGConvModule(
            in_channels=3,
            out_channels=self.in_planes,
            kernel_size=3,
            stride=2,
            padding=1,
            activation=activation,
            deploy=self.deploy,
        )
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(
            int(64 * width_multiplier[0]), num_blocks[0], stride=2
        )
        self.stage2 = self._make_stage(
            int(128 * width_multiplier[1]), num_blocks[1], stride=2
        )
        self.stage3 = self._make_stage(
            int(256 * width_multiplier[2]), num_blocks[2], stride=2
        )
        out_planes = last_channel if last_channel else int(512 * width_multiplier[3])
        self.stage4 = self._make_stage(out_planes, num_blocks[3], stride=2)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(
                RepVGGConvModule(
                    in_channels=self.in_planes,
                    out_channels=planes,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=cur_groups,
                    activation=self.activation,
                    deploy=self.deploy,
                )
            )
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stage0(x)
        output = []
        for i in range(1, 5):
            stage = getattr(self, "stage{}".format(i))
            x = stage(x)
            if i in self.out_stages:
                output.append(x)
        return tuple(output)


def repvgg_model_convert(model, deploy_model, save_path=None):
    """
    Examples:
        >>> train_model = RepVGG(arch='A0', deploy=False)
        >>> deploy_model = RepVGG(arch='A0', deploy=True)
        >>> deploy_model = repvgg_model_convert(
        >>>     train_model, deploy_model, save_path='repvgg_deploy.pth')
    """
    converted_weights = {}
    for name, module in model.named_modules():
        if hasattr(module, "repvgg_convert"):
            kernel, bias = module.repvgg_convert()
            converted_weights[name + ".rbr_reparam.weight"] = kernel
            converted_weights[name + ".rbr_reparam.bias"] = bias
        elif isinstance(module, torch.nn.Linear):
            converted_weights[name + ".weight"] = module.weight.detach().cpu().numpy()
            converted_weights[name + ".bias"] = module.bias.detach().cpu().numpy()
    del model

    for name, param in deploy_model.named_parameters():
        print("deploy param: ", name, param.size(), np.mean(converted_weights[name]))
        param.data = torch.from_numpy(converted_weights[name]).float()

    if save_path is not None:
        torch.save(deploy_model.state_dict(), save_path)

    return deploy_model


def repvgg_det_model_convert(model, deploy_model):
    converted_weights = {}
    deploy_model.load_state_dict(model.state_dict(), strict=False)
    for name, module in model.backbone.named_modules():
        if hasattr(module, "repvgg_convert"):
            kernel, bias = module.repvgg_convert()
            converted_weights[name + ".rbr_reparam.weight"] = kernel
            converted_weights[name + ".rbr_reparam.bias"] = bias
        elif isinstance(module, torch.nn.Linear):
            converted_weights[name + ".weight"] = module.weight.detach().cpu().numpy()
            converted_weights[name + ".bias"] = module.bias.detach().cpu().numpy()
    del model
    for name, param in deploy_model.backbone.named_parameters():
        print("deploy param: ", name, param.size(), np.mean(converted_weights[name]))
        param.data = torch.from_numpy(converted_weights[name]).float()
    return deploy_model
