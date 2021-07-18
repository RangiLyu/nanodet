import tempfile

from nanodet.data.dataset import build_dataset
from nanodet.evaluator import build_evaluator


def test_coco_detection():
    dummy_results = {
        0: {0: [[0, 0, 20, 20, 1]], 1: [[0, 0, 20, 20, 1]]},
        1: {0: [[0, 0, 20, 20, 1]]},
    }

    cfg = dict(
        name="CocoDataset",
        img_path="./tests/data",
        ann_path="./tests/data/dummy_coco.json",
        input_size=[320, 320],  # [w,h]
        keep_ratio=True,
        pipeline=dict(normalize=[[103.53, 116.28, 123.675], [57.375, 57.12, 58.395]]),
    )
    dataset = build_dataset(cfg, "train")

    eval_cfg = dict(name="CocoDetectionEvaluator", save_key="mAP")

    evaluator = build_evaluator(eval_cfg, dataset)
    tmp_dir = tempfile.TemporaryDirectory()
    eval_results = evaluator.evaluate(
        results=dummy_results, save_dir=tmp_dir.name, rank=-1
    )
    assert eval_results["mAP"] == 1
