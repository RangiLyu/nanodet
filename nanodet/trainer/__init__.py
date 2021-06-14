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

import torch

from .dist_trainer import DistTrainer
from .trainer import Trainer


def build_trainer(rank, cfg, model, logger):
    if len(cfg.device.gpu_ids) > 1:
        trainer = DistTrainer(rank, cfg, model, logger)
        trainer.set_device(cfg.device.batchsize_per_gpu,
                           rank,
                           device=torch.device("cuda"))  # TODO: device
    else:
        trainer = Trainer(rank, cfg, model, logger)
        trainer.set_device(
            cfg.device.batchsize_per_gpu,
            cfg.device.gpu_ids,
            device=torch.device("cuda"),
        )
    return trainer
