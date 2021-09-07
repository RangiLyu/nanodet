import pytest
import torch

from nanodet.data.batch_process import stack_batch_img


def test_stack_batch_img():
    with pytest.raises(AssertionError):
        dummy_imgs = [torch.rand((1, 30, 30)), torch.rand((3, 28, 10))]
        stack_batch_img(dummy_imgs, divisible=32, pad_value=0)

    dummy_imgs = [
        torch.rand((3, 300, 300)),
        torch.rand((3, 288, 180)),
        torch.rand((3, 169, 256)),
    ]
    batch_tensor = stack_batch_img(dummy_imgs, divisible=32, pad_value=0)
    assert batch_tensor.shape == (3, 3, 320, 320)
