import torch

from nanodet.model.module.scale import Scale


def test_scale():
    # test default scale
    scale = Scale()
    assert scale.scale.data == 1.0
    assert scale.scale.dtype == torch.float
    x = torch.rand(1, 3, 64, 64)
    output = scale(x)
    assert output.shape == (1, 3, 64, 64)

    # test given scale
    scale = Scale(10.0)
    assert scale.scale.data == 10.0
    assert scale.scale.dtype == torch.float
    x = torch.rand(1, 3, 64, 64)
    output = scale(x)
    assert output.shape == (1, 3, 64, 64)
