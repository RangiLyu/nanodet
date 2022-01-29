# Copyright 2022 RangiLyu.
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

import logging

import torch.nn as nn

logger = logging.getLogger("NanoDet")


class TIMMWrapper(nn.Module):
    """Wrapper to use backbones in timm
    https://github.com/rwightman/pytorch-image-models."""

    def __init__(
        self,
        model_name,
        features_only=True,
        pretrained=True,
        checkpoint_path="",
        in_channels=3,
        **kwargs,
    ):
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError(
                "timm is not installed, please install it first"
            ) from exc
        super(TIMMWrapper, self).__init__()
        self.timm = timm.create_model(
            model_name=model_name,
            features_only=features_only,
            pretrained=pretrained,
            in_chans=in_channels,
            checkpoint_path=checkpoint_path,
            **kwargs,
        )

        # Remove unused layers
        self.timm.global_pool = None
        self.timm.fc = None
        self.timm.classifier = None

        feature_info = getattr(self.timm, "feature_info", None)
        if feature_info:
            logger.info(f"TIMM backbone feature channels: {feature_info.channels()}")

    def forward(self, x):
        outs = self.timm(x)
        if isinstance(outs, (list, tuple)):
            features = tuple(outs)
        else:
            features = (outs,)
        return features
