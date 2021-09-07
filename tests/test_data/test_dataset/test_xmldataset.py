import pytest

from nanodet.data.dataset import XMLDataset, build_dataset


def test_xmldataset():
    class_names = ["asuka", "head"]
    cfg = dict(
        name="XMLDataset",
        class_names=class_names,
        img_path="./tests/data",
        ann_path="./tests/data",
        input_size=[320, 320],  # [w,h]
        keep_ratio=True,
        pipeline=dict(normalize=[[103.53, 116.28, 123.675], [57.375, 57.12, 58.395]]),
    )
    dataset = build_dataset(cfg, "train")
    assert isinstance(dataset, XMLDataset)

    for i, data in enumerate(dataset):
        assert data["img"].shape == (3, 320, 285)

    dataset = build_dataset(cfg, "val")
    for i, data in enumerate(dataset):
        assert data["img"].shape == (3, 320, 285)

    with pytest.raises(AssertionError):
        build_dataset(cfg, "2333")
