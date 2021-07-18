import math

import torch
import torch.functional as F
import torch.utils.model_zoo as model_zoo
from torch import nn

from ..module.activation import act_layers

efficientnet_lite_params = {
    # width_coefficient, depth_coefficient, image_size, dropout_rate
    "efficientnet_lite0": [1.0, 1.0, 224, 0.2],
    "efficientnet_lite1": [1.0, 1.1, 240, 0.2],
    "efficientnet_lite2": [1.1, 1.2, 260, 0.3],
    "efficientnet_lite3": [1.2, 1.4, 280, 0.3],
    "efficientnet_lite4": [1.4, 1.8, 300, 0.3],
}

model_urls = {
    "efficientnet_lite0": "https://github.com/RangiLyu/EfficientNet-Lite/releases/download/v1.0/efficientnet_lite0.pth",  # noqa: E501
    "efficientnet_lite1": "https://github.com/RangiLyu/EfficientNet-Lite/releases/download/v1.0/efficientnet_lite1.pth",  # noqa: E501
    "efficientnet_lite2": "https://github.com/RangiLyu/EfficientNet-Lite/releases/download/v1.0/efficientnet_lite2.pth",  # noqa: E501
    "efficientnet_lite3": "https://github.com/RangiLyu/EfficientNet-Lite/releases/download/v1.0/efficientnet_lite3.pth",  # noqa: E501
    "efficientnet_lite4": "https://github.com/RangiLyu/EfficientNet-Lite/releases/download/v1.0/efficientnet_lite4.pth",  # noqa: E501
}


def round_filters(filters, multiplier, divisor=8, min_width=None):
    """Calculate and round number of filters based on width multiplier."""
    if not multiplier:
        return filters
    filters *= multiplier
    min_width = min_width or divisor
    new_filters = max(min_width, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, multiplier):
    """Round number of filters based on depth multiplier."""
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def drop_connect(x, drop_connect_rate, training):
    if not training:
        return x
    keep_prob = 1.0 - drop_connect_rate
    batch_size = x.shape[0]
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=x.dtype, device=x.device)
    binary_mask = torch.floor(random_tensor)
    x = (x / keep_prob) * binary_mask
    return x


class MBConvBlock(nn.Module):
    def __init__(
        self,
        inp,
        final_oup,
        k,
        s,
        expand_ratio,
        se_ratio,
        has_se=False,
        activation="ReLU6",
    ):
        super(MBConvBlock, self).__init__()

        self._momentum = 0.01
        self._epsilon = 1e-3
        self.input_filters = inp
        self.output_filters = final_oup
        self.stride = s
        self.expand_ratio = expand_ratio
        self.has_se = has_se
        self.id_skip = True  # skip connection and drop connect

        # Expansion phase
        oup = inp * expand_ratio  # number of output channels
        if expand_ratio != 1:
            self._expand_conv = nn.Conv2d(
                in_channels=inp, out_channels=oup, kernel_size=1, bias=False
            )
            self._bn0 = nn.BatchNorm2d(
                num_features=oup, momentum=self._momentum, eps=self._epsilon
            )

        # Depthwise convolution phase
        self._depthwise_conv = nn.Conv2d(
            in_channels=oup,
            out_channels=oup,
            groups=oup,  # groups makes it depthwise
            kernel_size=k,
            padding=(k - 1) // 2,
            stride=s,
            bias=False,
        )
        self._bn1 = nn.BatchNorm2d(
            num_features=oup, momentum=self._momentum, eps=self._epsilon
        )

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(inp * se_ratio))
            self._se_reduce = nn.Conv2d(
                in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1
            )
            self._se_expand = nn.Conv2d(
                in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1
            )

        # Output phase
        self._project_conv = nn.Conv2d(
            in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False
        )
        self._bn2 = nn.BatchNorm2d(
            num_features=final_oup, momentum=self._momentum, eps=self._epsilon
        )
        self._relu = act_layers(activation)

    def forward(self, x, drop_connect_rate=None):
        """
        :param x: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        identity = x
        if self.expand_ratio != 1:
            x = self._relu(self._bn0(self._expand_conv(x)))
        x = self._relu(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._relu(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        if (
            self.id_skip
            and self.stride == 1
            and self.input_filters == self.output_filters
        ):
            if drop_connect_rate:
                x = drop_connect(x, drop_connect_rate, training=self.training)
            x += identity  # skip connection
        return x


class EfficientNetLite(nn.Module):
    def __init__(
        self, model_name, out_stages=(2, 4, 6), activation="ReLU6", pretrain=True
    ):
        super(EfficientNetLite, self).__init__()
        assert set(out_stages).issubset(i for i in range(0, 7))
        assert model_name in efficientnet_lite_params

        self.model_name = model_name
        # Batch norm parameters
        momentum = 0.01
        epsilon = 1e-3
        width_multiplier, depth_multiplier, _, dropout_rate = efficientnet_lite_params[
            model_name
        ]
        self.drop_connect_rate = 0.2
        self.out_stages = out_stages

        mb_block_settings = [
            # repeat|kernel_size|stride|expand|input|output|se_ratio
            [1, 3, 1, 1, 32, 16, 0.25],  # stage0
            [2, 3, 2, 6, 16, 24, 0.25],  # stage1 - 1/4
            [2, 5, 2, 6, 24, 40, 0.25],  # stage2 - 1/8
            [3, 3, 2, 6, 40, 80, 0.25],  # stage3
            [3, 5, 1, 6, 80, 112, 0.25],  # stage4 - 1/16
            [4, 5, 2, 6, 112, 192, 0.25],  # stage5
            [1, 3, 1, 6, 192, 320, 0.25],  # stage6 - 1/32
        ]

        # Stem
        out_channels = 32
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels, momentum=momentum, eps=epsilon),
            act_layers(activation),
        )

        # Build blocks
        self.blocks = nn.ModuleList([])
        for i, stage_setting in enumerate(mb_block_settings):
            stage = nn.ModuleList([])
            (
                num_repeat,
                kernal_size,
                stride,
                expand_ratio,
                input_filters,
                output_filters,
                se_ratio,
            ) = stage_setting
            # Update block input and output filters based on width multiplier.
            input_filters = (
                input_filters
                if i == 0
                else round_filters(input_filters, width_multiplier)
            )
            output_filters = round_filters(output_filters, width_multiplier)
            num_repeat = (
                num_repeat
                if i == 0 or i == len(mb_block_settings) - 1
                else round_repeats(num_repeat, depth_multiplier)
            )

            # The first block needs to take care of stride and filter size increase.
            stage.append(
                MBConvBlock(
                    input_filters,
                    output_filters,
                    kernal_size,
                    stride,
                    expand_ratio,
                    se_ratio,
                    has_se=False,
                )
            )
            if num_repeat > 1:
                input_filters = output_filters
                stride = 1
            for _ in range(num_repeat - 1):
                stage.append(
                    MBConvBlock(
                        input_filters,
                        output_filters,
                        kernal_size,
                        stride,
                        expand_ratio,
                        se_ratio,
                        has_se=False,
                    )
                )

            self.blocks.append(stage)
        self._initialize_weights(pretrain)

    def forward(self, x):
        x = self.stem(x)
        output = []
        idx = 0
        for j, stage in enumerate(self.blocks):
            for block in stage:
                drop_connect_rate = self.drop_connect_rate
                if drop_connect_rate:
                    drop_connect_rate *= float(idx) / len(self.blocks)
                x = block(x, drop_connect_rate)
                idx += 1
            if j in self.out_stages:
                output.append(x)
        return output

    def _initialize_weights(self, pretrain=True):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if pretrain:
            url = model_urls[self.model_name]
            if url is not None:
                pretrained_state_dict = model_zoo.load_url(url)
                print("=> loading pretrained model {}".format(url))
                self.load_state_dict(pretrained_state_dict, strict=False)

    def load_pretrain(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict, strict=True)
