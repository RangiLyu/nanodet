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

from .nanodet_plus import NanoDetPlus
from .one_stage_detector import OneStageDetector


def build_model(model_cfg):
    model_cfg = copy.deepcopy(model_cfg)
    name = model_cfg.arch.pop("name")
    if name == "GFL":
        warnings.warn(
            "Model architecture name is changed to 'OneStageDetector'. "
            "The name 'GFL' is deprecated, please change the model->arch->name "
            "in your YAML config file to OneStageDetector."
        )
        model = OneStageDetector(
            model_cfg.arch.backbone, model_cfg.arch.fpn, model_cfg.arch.head
        )
    elif name == "OneStageDetector":
        model = OneStageDetector(
            model_cfg.arch.backbone, model_cfg.arch.fpn, model_cfg.arch.head
        )
    elif name == "NanoDetPlus":
        model = NanoDetPlus(**model_cfg.arch)
    else:
        raise NotImplementedError
    return model
