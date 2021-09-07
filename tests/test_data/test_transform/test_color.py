import numpy as np

from nanodet.data.transform.color import (
    normalize,
    random_brightness,
    random_contrast,
    random_saturation,
)


def test_random_color_aug():
    img = np.ones((10, 10, 3), dtype=np.float32)

    res = random_brightness(img, 0.6)
    assert np.max(res) <= 1.6
    assert np.min(res) >= 0.4

    img = np.ones((10, 10, 3), dtype=np.float32)
    res = random_contrast(img, 0.5, 1.5)
    assert np.max(res) <= 1.5
    assert np.min(res) >= 0.5

    img = np.ones((10, 10, 3), dtype=np.float32)
    random_saturation(img, 0.5, 1.5)

    img = np.ones((10, 10, 3), dtype=np.uint8) * 255
    meta = dict(img=img)
    res = normalize(meta, [100, 100, 100], [155, 155, 155])
    assert np.array_equal(res["img"], np.ones((10, 10, 3)))
