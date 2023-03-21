import numpy as np

from nanodet.data.collate import naive_collate


def test_naive_collate():
    batch = [
        {"bbox": np.zeros((5, 4))},
        {"bbox": np.zeros((3, 4))},
        {"bbox": np.zeros((6, 4))},
    ]
    batch = naive_collate(batch)
    assert isinstance(batch, dict)
    assert isinstance(batch["bbox"], list)
    assert len(batch["bbox"]) == 3
    assert batch["bbox"][0].shape == (5, 4)
    assert batch["bbox"][1].shape == (3, 4)
    assert batch["bbox"][2].shape == (6, 4)
