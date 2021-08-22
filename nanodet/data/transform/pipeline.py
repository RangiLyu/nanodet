# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import warnings
from typing import Dict, Tuple

from torch.utils.data import Dataset

from .color import color_aug_and_norm
from .warp import ShapeTransform, warp_and_resize


class LegacyPipeline:
    def __init__(self, cfg, keep_ratio):
        warnings.warn(
            "Deprecated warning! Pipeline from nanodet v0.x has been deprecated,"
            "Please use new Pipeline and update your config!"
        )
        self.warp = functools.partial(
            warp_and_resize, warp_kwargs=cfg, keep_ratio=keep_ratio
        )
        self.color = functools.partial(color_aug_and_norm, kwargs=cfg)

    def __call__(self, meta, dst_shape):
        meta = self.warp(meta, dst_shape=dst_shape)
        meta = self.color(meta=meta)
        return meta


class Pipeline:
    """Data process pipeline. Apply augmentation and pre-processing on
    meta_data from dataset.

    Args:
        cfg (Dict): Data pipeline config.
        keep_ratio (bool): Whether to keep aspect ratio when resizing image.

    """

    def __init__(self, cfg: Dict, keep_ratio: bool):
        self.shape_transform = ShapeTransform(keep_ratio, **cfg)
        self.color = functools.partial(color_aug_and_norm, kwargs=cfg)

    def __call__(self, dataset: Dataset, meta: Dict, dst_shape: Tuple[int, int]):
        meta = self.shape_transform(meta, dst_shape=dst_shape)
        meta = self.color(meta=meta)
        return meta
