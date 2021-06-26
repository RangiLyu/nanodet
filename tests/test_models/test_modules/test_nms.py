import pickle

import torch

from nanodet.model.module.nms import batched_nms, multiclass_nms


def test_batched_nms():
    file = open("./tests/data/batched_nms_data.pkl", "rb")
    results = pickle.load(file)

    nms_cfg = dict(iou_threshold=0.7)
    boxes, keep = batched_nms(
        torch.from_numpy(results["boxes"]),
        torch.from_numpy(results["scores"]),
        torch.from_numpy(results["idxs"]),
        nms_cfg,
        class_agnostic=False,
    )

    nms_cfg.update(split_thr=100)
    seq_boxes, seq_keep = batched_nms(
        torch.from_numpy(results["boxes"]),
        torch.from_numpy(results["scores"]),
        torch.from_numpy(results["idxs"]),
        nms_cfg,
        class_agnostic=False,
    )

    assert torch.equal(keep, seq_keep)
    assert torch.equal(boxes, seq_boxes)


def test_multiclass_nms():
    file = open("./tests/data/batched_nms_data.pkl", "rb")
    results = pickle.load(file)
    det_boxes = torch.from_numpy(results["boxes"])

    socres = torch.rand(det_boxes.shape[0], 8)
    score_thr = 0.9
    max_num = 100
    nms_cfg = dict(iou_threshold=0.5)
    boxes, keep = multiclass_nms(
        det_boxes, socres, score_thr=score_thr, nms_cfg=nms_cfg, max_num=max_num
    )

    assert boxes.shape[0] <= max_num
    assert keep.shape[0] <= max_num
    assert min(boxes[:, -1]) >= score_thr

    # test all zero score
    socres = torch.zeros(det_boxes.shape[0], 8)
    score_thr = 0.1
    nms_cfg = dict(iou_threshold=0.5)
    boxes, keep = multiclass_nms(
        det_boxes, socres, score_thr=score_thr, nms_cfg=nms_cfg, max_num=-1
    )
    assert boxes.shape[0] == 0
    assert keep.shape[0] == 0
