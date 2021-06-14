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

import torch.distributed as dist

from ..util import DDP
from .trainer import Trainer


def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size


class DistTrainer(Trainer):
    """
    Distributed trainer for multi-gpu training. (not finish yet)
    """

    def run_step(self, model, batch, mode='train'):
        output, loss, loss_stats = model.module.forward_train(batch)
        loss = loss.mean()
        if mode == 'train':
            self.optimizer.zero_grad()
            loss.backward()
            average_gradients(model)
            self.optimizer.step()
        return output, loss, loss_stats

    def set_device(self, batch_per_gpu, rank, device):
        """
        Set model device for Distributed-Data-Parallel
        :param batch_per_gpu: batch size of each gpu
        :param rank: distributed training process rank
        :param device: cuda
        """
        self.rank = rank
        self.model = DDP(batch_per_gpu, module=self.model.cuda(), device_ids=[rank], output_device=rank)
