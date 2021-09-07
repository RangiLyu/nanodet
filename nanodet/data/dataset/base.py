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
import random
from abc import ABCMeta, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset

from ..transform import Pipeline


class BaseDataset(Dataset, metaclass=ABCMeta):
    """
    A base class of detection dataset. Referring from MMDetection.
    A dataset should have images, annotations and preprocessing pipelines
    NanoDet use [xmin, ymin, xmax, ymax] format for box and
     [[x0,y0], [x1,y1] ... [xn,yn]] format for key points.
    instance masks should decode into binary masks for each instance like
    {
        'bbox': [xmin,ymin,xmax,ymax],
        'mask': mask
     }
    segmentation mask should decode into binary masks for each class.
    Args:
        img_path (str): image data folder
        ann_path (str): annotation file path or folder
        use_instance_mask (bool): load instance segmentation data
        use_seg_mask (bool): load semantic segmentation data
        use_keypoint (bool): load pose keypoint data
        load_mosaic (bool): using mosaic data augmentation from yolov4
        mode (str): 'train' or 'val' or 'test'
        multi_scale (Tuple[float, float]): Multi-scale factor range.
    """

    def __init__(
        self,
        img_path: str,
        ann_path: str,
        input_size: Tuple[int, int],
        pipeline: Dict,
        keep_ratio: bool = True,
        use_instance_mask: bool = False,
        use_seg_mask: bool = False,
        use_keypoint: bool = False,
        load_mosaic: bool = False,
        mode: str = "train",
        multi_scale: Optional[Tuple[float, float]] = None,
    ):
        assert mode in ["train", "val", "test"]
        self.img_path = img_path
        self.ann_path = ann_path
        self.input_size = input_size
        self.pipeline = Pipeline(pipeline, keep_ratio)
        self.keep_ratio = keep_ratio
        self.use_instance_mask = use_instance_mask
        self.use_seg_mask = use_seg_mask
        self.use_keypoint = use_keypoint
        self.load_mosaic = load_mosaic
        self.multi_scale = multi_scale
        self.mode = mode

        self.data_info = self.get_data_info(ann_path)

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        if self.mode == "val" or self.mode == "test":
            return self.get_val_data(idx)
        else:
            while True:
                data = self.get_train_data(idx)
                if data is None:
                    idx = self.get_another_id()
                    continue
                return data

    @staticmethod
    def get_random_size(
        scale_range: Tuple[float, float], image_size: Tuple[int, int]
    ) -> Tuple[int, int]:
        """
        Get random image shape by multi-scale factor and image_size.
        Args:
            scale_range (Tuple[float, float]): Multi-scale factor range.
                Format in [(width, height), (width, height)]
            image_size (Tuple[int, int]): Image size. Format in (width, height).

        Returns:
            Tuple[int, int]
        """
        assert len(scale_range) == 2
        scale_factor = random.uniform(*scale_range)
        width = int(image_size[0] * scale_factor)
        height = int(image_size[1] * scale_factor)
        return width, height

    @abstractmethod
    def get_data_info(self, ann_path):
        pass

    @abstractmethod
    def get_train_data(self, idx):
        pass

    @abstractmethod
    def get_val_data(self, idx):
        pass

    def get_another_id(self):
        return np.random.random_integers(0, len(self.data_info) - 1)
