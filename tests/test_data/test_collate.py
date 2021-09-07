import numpy as np
import torch

from nanodet.data.collate import collate_function


def test_collate():
    batch = [1.2, 2.3, 3.4]
    out = collate_function(batch)
    assert isinstance(out, torch.Tensor)

    batch = [1, 2, 3]
    out = collate_function(batch)
    assert isinstance(out, torch.Tensor)

    batch = ["1", "2", "3"]
    out = collate_function(batch)
    assert out == batch

    batch = [{"1": 1, "2": 1.2, "3": 1.2}, {"1": 2, "2": 1.3, "3": 1.4}]
    out = collate_function(batch)
    assert isinstance(out, dict)
    for k, v in out.items():
        assert isinstance(v, torch.Tensor)

    batch = [np.array([1, 2, 3]), np.array([4, 6, 8])]
    out = collate_function(batch)
    assert out == batch

    batch = [torch.randn((3, 20, 20)), torch.randn((3, 20, 20))]
    out = collate_function(batch)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 3, 20, 20)
