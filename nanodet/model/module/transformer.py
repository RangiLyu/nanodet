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

import torch.nn as nn

from nanodet.model.module.activation import act_layers
from nanodet.model.module.conv import ConvModule


class MLP(nn.Module):
    def __init__(
        self, in_dim, hidden_dim=None, out_dim=None, drop=0.0, activation="GELU"
    ):
        super(MLP, self).__init__()
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = act_layers(activation)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerEncoder(nn.Module):
    """
    Encoder layer of transformer
    :param dim: feature dimension
    :param num_heads: number of attention heads
    :param mlp_ratio: hidden layer dimension expand ratio in MLP
    :param dropout_ratio: probability of an element to be zeroed
    :param activation: activation layer type
    :param kv_bias: add bias on key and values
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio,
        dropout_ratio=0.0,
        activation="GELU",
        kv_bias=False,
    ):
        super(TransformerEncoder, self).__init__()
        self.norm1 = nn.LayerNorm(dim)

        # embed_dim must be divisible by num_heads
        assert dim // num_heads * num_heads == dim
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout_ratio,
            add_bias_kv=kv_bias,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(
            in_dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            drop=dropout_ratio,
            activation=activation,
        )

    def forward(self, x):
        _x = self.norm1(x)
        x = x + self.attn(_x, _x, _x)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerBlock(nn.Module):
    """
    Block of transformer encoder layers. Used in vision task.
    :param in_channels: input channels
    :param out_channels: output channels
    :param num_heads: number of attention heads
    :param num_encoders: number of transformer encoder layers
    :param mlp_ratio: hidden layer dimension expand ratio in MLP
    :param dropout_ratio: probability of an element to be zeroed
    :param activation: activation layer type
    :param kv_bias: add bias on key and values
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_heads,
        num_encoders=1,
        mlp_ratio=1,
        dropout_ratio=0.0,
        kv_bias=False,
        activation="GELU",
    ):
        super(TransformerBlock, self).__init__()

        # out_channels must be divisible by num_heads
        assert out_channels // num_heads * num_heads == out_channels

        self.conv = (
            nn.Identity()
            if in_channels == out_channels
            else ConvModule(in_channels, out_channels, 1)
        )
        self.linear = nn.Linear(out_channels, out_channels)
        encoders = [
            TransformerEncoder(
                out_channels, num_heads, mlp_ratio, dropout_ratio, activation, kv_bias
            )
            for _ in range(num_encoders)
        ]
        self.encoders = nn.Sequential(*encoders)

    def forward(self, x, pos_embed):
        b, _, h, w = x.shape
        x = self.conv(x)
        x = x.flatten(2).permute(2, 0, 1)
        x = x + pos_embed
        x = self.encoders(x)
        x = x.permute(1, 2, 0).reshape(b, -1, h, w)
        return x
