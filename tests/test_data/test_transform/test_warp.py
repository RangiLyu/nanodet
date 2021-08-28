import copy

import numpy as np

from nanodet.data.transform.warp import (
    ShapeTransform,
    get_flip_matrix,
    get_perspective_matrix,
    get_rotation_matrix,
    get_scale_matrix,
    get_shear_matrix,
    get_stretch_matrix,
    get_translate_matrix,
    warp_and_resize,
)


def test_get_matrix():
    # TODO: better unit test
    height = 100
    width = 200

    # center
    C = np.eye(3)
    C[0, 2] = -width / 2
    C[1, 2] = -height / 2

    # do not change the order of mat mul
    P = get_perspective_matrix(0.1)
    C = P @ C

    Scl = get_scale_matrix((1, 2))
    C = Scl @ C

    Str = get_stretch_matrix((0.5, 1.5), (0.5, 1.5))
    C = Str @ C

    R = get_rotation_matrix(180)
    C = R @ C

    Sh = get_shear_matrix(60)
    C = Sh @ C

    F = get_flip_matrix(0.5)
    C = F @ C

    T = get_translate_matrix(0.5, width, height)

    M = T @ C

    assert M.shape == (3, 3)


def test_warp():
    dummy_meta = dict(
        img=np.random.randint(0, 255, size=(100, 200, 3), dtype=np.uint8),
        gt_bboxes=np.array([[0, 0, 20, 20]]),
        gt_masks=[np.zeros((100, 200), dtype=np.uint8)],
    )
    warp_cfg = {}
    res = warp_and_resize(
        copy.deepcopy(dummy_meta), warp_cfg, dst_shape=(50, 50), keep_ratio=False
    )
    assert res["img"].shape == (50, 50, 3)
    assert res["gt_masks"][0].shape == (50, 50)
    assert np.array_equal(res["gt_bboxes"], np.array([[0, 0, 5, 10]], dtype=np.float32))

    res = warp_and_resize(
        copy.deepcopy(dummy_meta), warp_cfg, dst_shape=(50, 50), keep_ratio=True
    )
    assert np.array_equal(
        res["gt_bboxes"], np.array([[0, 12.5, 5.0, 17.5]], dtype=np.float32)
    )

    res = warp_and_resize(
        copy.deepcopy(dummy_meta), warp_cfg, dst_shape=(300, 300), keep_ratio=True
    )
    assert np.array_equal(
        res["gt_bboxes"], np.array([[0, 75, 30, 105]], dtype=np.float32)
    )


def test_shape_transform():
    dummy_meta = dict(
        img=np.random.randint(0, 255, size=(100, 200, 3), dtype=np.uint8),
        gt_bboxes=np.array([[0, 0, 20, 20]]),
        gt_masks=[np.zeros((100, 200), dtype=np.uint8)],
    )
    # keep ratio
    transform = ShapeTransform(keep_ratio=True, divisible=32)
    res = transform(dummy_meta, dst_shape=(50, 50))
    assert np.array_equal(
        res["gt_bboxes"], np.array([[0, 0, 6.4, 6.4]], dtype=np.float32)
    )
    assert res["img"].shape[0] % 32 == 0
    assert res["img"].shape[1] % 32 == 0

    # not keep ratio
    transform = ShapeTransform(keep_ratio=False)
    res = transform(dummy_meta, dst_shape=(50, 50))
    assert np.array_equal(res["gt_bboxes"], np.array([[0, 0, 5, 10]], dtype=np.float32))
