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

import torch

from ..head import build_head
from .one_stage_detector import OneStageDetector


class NanoDetPlus(OneStageDetector):
    def __init__(
        self,
        backbone,
        fpn,
        aux_head,
        head,
        detach_epoch=0,
    ):
        super(NanoDetPlus, self).__init__(
            backbone_cfg=backbone, fpn_cfg=fpn, head_cfg=head
        )
        self.aux_fpn = copy.deepcopy(self.fpn)
        self.aux_head = build_head(aux_head)
        self.detach_epoch = detach_epoch

    def forward_train(self, gt_meta):
        img = gt_meta["img"]
        feat = self.backbone(img)
        fpn_feat = self.fpn(feat)
        if self.epoch >= self.detach_epoch:
            aux_fpn_feat = self.aux_fpn([f.detach() for f in feat])
            dual_fpn_feat = (
                torch.cat([f.detach(), aux_f], dim=1)
                for f, aux_f in zip(fpn_feat, aux_fpn_feat)
            )
        else:
            aux_fpn_feat = self.aux_fpn(feat)
            dual_fpn_feat = (
                torch.cat([f, aux_f], dim=1) for f, aux_f in zip(fpn_feat, aux_fpn_feat)
            )
        head_out = self.head(fpn_feat)
        aux_head_out = self.aux_head(dual_fpn_feat)
        loss, loss_states = self.head.loss(head_out, gt_meta, aux_preds=aux_head_out)
        return head_out, loss, loss_states
