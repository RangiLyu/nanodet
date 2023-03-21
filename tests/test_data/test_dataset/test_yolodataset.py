from nanodet.data.dataset import YoloDataset


def test_yolodataset():
    ann_path = "tests/data"
    yolodataset = YoloDataset(
        img_path=ann_path,
        ann_path=ann_path,
        class_names=["class1"],
        input_size=[320, 320],  # [w,h]
        keep_ratio=False,
        pipeline=dict(normalize=[[103.53, 116.28, 123.675], [57.375, 57.12, 58.395]]),
    )
    assert len(yolodataset) == 1
    for i, data in enumerate(yolodataset):
        assert data["img"].shape == (3, 320, 320)
        assert data["gt_bboxes"].shape == (6, 4)
