# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..module.conv import ConvModule
from ..module.transformer import TransformerBlock
from ..module.init_weights import normal_init


class TAN(nn.Module):
    """
    Transformer Attention Network.

    :param in_channels: Number of input channels per scale.
    :param out_channels: Number of output channel.
    :param feature_hw: Size of feature map input to transformer.
    :param num_heads: Number of attention heads.
    :param num_encoders: Number of transformer encoder layers.
    :param mlp_ratio: Hidden layer dimension expand ratio in MLP.
    :param dropout_ratio: Probability of an element to be zeroed.
    :param activation: Activation layer type.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 feature_hw,
                 num_heads,
                 num_encoders,
                 mlp_ratio,
                 dropout_ratio,
                 activation='LeakyReLU'
                 ):
        super(TAN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        assert self.num_ins == 3

        self.lateral_convs = nn.ModuleList()
        for i in range(self.num_ins):
            l_conv = ConvModule(in_channels[i], out_channels, 1,
                norm_cfg=dict(type='BN'), activation=activation, inplace=False)
            self.lateral_convs.append(l_conv)
        self.transformer = TransformerBlock(out_channels*self.num_ins, out_channels, num_heads,
                                            num_encoders, mlp_ratio, dropout_ratio, activation=activation)
        self.pos_embed = nn.Parameter(torch.zeros(feature_hw[0] * feature_hw[1], 1, out_channels))

        self.init_weights()

    def init_weights(self):
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                normal_init(m, 0.01)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # transformer attention
        mid_shape = laterals[1].shape[2:]
        mid_lvl = torch.cat((F.interpolate(laterals[0], size=mid_shape, mode='bilinear'),
                  laterals[1], F.interpolate(laterals[2], size=mid_shape, mode='bilinear')), dim=1)
        mid_lvl = self.transformer(mid_lvl, self.pos_embed)

        # outs = [
        #     torch.cat((laterals[0], F.interpolate(mid_lvl, size=laterals[0].shape[2:], mode='bilinear')), dim=1),
        #     torch.cat((laterals[1], mid_lvl), dim=1),
        #     torch.cat((laterals[2], F.interpolate(mid_lvl, size=laterals[2].shape[2:], mode='bilinear')), dim=1)
        # ]
        # build outputs
        outs = [
            laterals[0]+F.interpolate(mid_lvl, size=laterals[0].shape[2:], mode='bilinear'),
            laterals[1]+mid_lvl,
            laterals[2]+F.interpolate(mid_lvl, size=laterals[2].shape[2:], mode='bilinear')
        ]
        return tuple(outs)
