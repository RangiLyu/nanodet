import pytest
import torch
import torch.nn as nn

from nanodet.model.module.transformer import MLP, TransformerBlock, TransformerEncoder


def test_mlp():
    mlp = MLP(in_dim=10)
    assert mlp.fc1.in_features == 10
    assert mlp.fc1.out_features == 10
    assert mlp.fc2.in_features == 10
    assert mlp.fc2.out_features == 10

    mlp = MLP(in_dim=10, hidden_dim=5, out_dim=12, drop=0.2)
    assert mlp.fc1.in_features == 10
    assert mlp.fc1.out_features == 5
    assert mlp.fc2.in_features == 5
    assert mlp.fc2.out_features == 12
    assert mlp.drop.p == 0.2
    input = torch.rand(64, 10)
    output = mlp(input)
    assert output.shape == (64, 12)


def test_tansformer_encoder():
    # embed_dim must be divisible by num_heads
    with pytest.raises(AssertionError):
        TransformerEncoder(10, 6, 0.5)

    encoder = TransformerEncoder(10, 5, 1)
    assert isinstance(encoder.norm1, nn.LayerNorm)
    assert isinstance(encoder.norm2, nn.LayerNorm)
    assert encoder.attn.embed_dim == 10
    input = torch.rand(32, 1, 10)
    output = encoder(input)
    assert output.shape == (32, 1, 10)


def test_transformer_block():
    # out_channels must be divisible by num_heads
    with pytest.raises(AssertionError):
        TransformerBlock(10, 15, 4)

    # test in dim equal to out dim
    block = TransformerBlock(15, 15, 3)
    assert isinstance(block.conv, nn.Identity)

    block = TransformerBlock(12, 15, 3, 2)
    input = torch.rand(1, 12, 16, 16)
    pos_embed = torch.rand(16 * 16, 1, 15)
    out = block(input, pos_embed)
    assert out.shape == (1, 15, 16, 16)
