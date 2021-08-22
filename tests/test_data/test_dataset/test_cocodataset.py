import pytest

from nanodet.data.dataset import CocoDataset, build_dataset


def test_cocodataset():
    cfg = dict(
        name="CocoDataset",
        img_path="./tests/data",
        ann_path="./tests/data/dummy_coco.json",
        input_size=[320, 320],  # [w,h]
        keep_ratio=True,
        use_instance_mask=True,
        pipeline=dict(normalize=[[103.53, 116.28, 123.675], [57.375, 57.12, 58.395]]),
    )
    dataset = build_dataset(cfg, "train")
    assert isinstance(dataset, CocoDataset)

    for i, data in enumerate(dataset):
        assert data["img"].shape == (3, 320, 285)
        for mask in data["gt_masks"]:
            assert mask.shape == (320, 285)

    dataset = build_dataset(cfg, "val")
    for i, data in enumerate(dataset):
        assert data["img"].shape == (3, 320, 285)
        for mask in data["gt_masks"]:
            assert mask.shape == (320, 285)

    cfg["keep_ratio"] = False
    dataset = build_dataset(cfg, "train")
    for i, data in enumerate(dataset):
        assert data["img"].shape == (3, 320, 320)
        for mask in data["gt_masks"]:
            assert mask.shape == (320, 320)

    with pytest.raises(AssertionError):
        build_dataset(cfg, "2333")


def test_multi_scale():
    cfg = dict(
        name="CocoDataset",
        img_path="./tests/data",
        ann_path="./tests/data/dummy_coco.json",
        input_size=[320, 320],  # [w,h]
        multi_scale=[1.5, 1.5],
        keep_ratio=True,
        use_instance_mask=True,
        pipeline=dict(normalize=[[103.53, 116.28, 123.675], [57.375, 57.12, 58.395]]),
    )
    dataset = build_dataset(cfg, "train")

    for i, data in enumerate(dataset):
        assert data["img"].shape == (3, 480, 427)
        for mask in data["gt_masks"]:
            assert mask.shape == (480, 427)
