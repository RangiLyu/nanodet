from __future__ import absolute_import, division, print_function

import warnings

import torch.nn as nn

from ..module.activation import act_layers


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size=3,
        stride=1,
        groups=1,
        activation="ReLU",
    ):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            act_layers(activation),
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, activation="ReLU"):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(
                ConvBNReLU(inp, hidden_dim, kernel_size=1, activation=activation)
            )
        layers.extend(
            [
                # dw
                ConvBNReLU(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    activation=activation,
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
        self,
        width_mult=1.0,
        out_stages=(1, 2, 4, 6),
        last_channel=1280,
        activation="ReLU",
        act=None,
    ):
        super(MobileNetV2, self).__init__()
        # TODO: support load torchvison pretrained weight
        assert set(out_stages).issubset(i for i in range(7))
        self.width_mult = width_mult
        self.out_stages = out_stages
        input_channel = 32
        self.last_channel = last_channel
        self.activation = activation
        if act is not None:
            warnings.warn(
                "Warning! act argument has been deprecated, " "use activation instead!"
            )
            self.activation = act
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        self.input_channel = int(input_channel * width_mult)
        self.first_layer = ConvBNReLU(
            3, self.input_channel, stride=2, activation=self.activation
        )
        # building inverted residual blocks
        for i in range(7):
            name = "stage{}".format(i)
            setattr(self, name, self.build_mobilenet_stage(stage_num=i))

        self._initialize_weights()

    def build_mobilenet_stage(self, stage_num):
        stage = []
        t, c, n, s = self.interverted_residual_setting[stage_num]
        output_channel = int(c * self.width_mult)
        for i in range(n):
            if i == 0:
                stage.append(
                    InvertedResidual(
                        self.input_channel,
                        output_channel,
                        s,
                        expand_ratio=t,
                        activation=self.activation,
                    )
                )
            else:
                stage.append(
                    InvertedResidual(
                        self.input_channel,
                        output_channel,
                        1,
                        expand_ratio=t,
                        activation=self.activation,
                    )
                )
            self.input_channel = output_channel
        if stage_num == 6:
            last_layer = ConvBNReLU(
                self.input_channel,
                self.last_channel,
                kernel_size=1,
                activation=self.activation,
            )
            stage.append(last_layer)
        stage = nn.Sequential(*stage)
        return stage

    def forward(self, x):
        x = self.first_layer(x)
        output = []
        for i in range(0, 7):
            stage = getattr(self, "stage{}".format(i))
            x = stage(x)
            if i in self.out_stages:
                output.append(x)

        return tuple(output)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
