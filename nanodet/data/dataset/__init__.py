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

import copy
import warnings

from .coco import CocoDataset
from .xml_dataset import XMLDataset
from .yolo import YoloDataset


def build_dataset(cfg, mode):
    dataset_cfg = copy.deepcopy(cfg)
    name = dataset_cfg.pop("name")
    if name == "coco":
        warnings.warn(
            "Dataset name coco has been deprecated. Please use CocoDataset instead."
        )
        return CocoDataset(mode=mode, **dataset_cfg)
    elif name == "yolo":
        return YoloDataset(mode=mode, **dataset_cfg)
    elif name == "xml_dataset":
        warnings.warn(
            "Dataset name xml_dataset has been deprecated. "
            "Please use XMLDataset instead."
        )
        return XMLDataset(mode=mode, **dataset_cfg)
    elif name == "CocoDataset":
        return CocoDataset(mode=mode, **dataset_cfg)
    elif name == "YoloDataset":
        return YoloDataset(mode=mode, **dataset_cfg)
    elif name == "XMLDataset":
        return XMLDataset(mode=mode, **dataset_cfg)
    else:
        raise NotImplementedError("Unknown dataset type!")
