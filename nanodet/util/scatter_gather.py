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

import pickle

import torch
import torch.distributed as dist
from torch.autograd import Variable
from torch.nn.parallel._functions import Scatter


def list_scatter(input, target_gpus, chunk_sizes):
    ret = []
    for idx, size in enumerate(chunk_sizes):
        ret.append(input[:size])
        del input[:size]
    return tuple(ret)


def scatter(inputs, target_gpus, dim=0, chunk_sizes=None):
    """
    Slices variables into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not variables. Does not
    support Tensors.
    """

    def scatter_map(obj):
        if isinstance(obj, Variable):
            return Scatter.apply(target_gpus, chunk_sizes, dim, obj)
        assert not torch.is_tensor(obj), "Tensors not supported in scatter."
        if isinstance(obj, list):
            return list_scatter(obj, target_gpus, chunk_sizes)
        if isinstance(obj, tuple):
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, dict):
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]

    return scatter_map(inputs)


def scatter_kwargs(inputs, kwargs, target_gpus, dim=0, chunk_sizes=None):
    r"""Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, dim, chunk_sizes) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim, chunk_sizes) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


def gather_results(result_part):
    rank = -1
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device="cuda"
    )

    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device="cuda")
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)

    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device="cuda")
    part_send[: shape_tensor[0]] = part_tensor
    part_recv_list = [part_tensor.new_zeros(shape_max) for _ in range(world_size)]

    # gather all result dict
    dist.all_gather(part_recv_list, part_send)

    if rank < 1:
        all_res = {}
        for recv, shape in zip(part_recv_list, shape_list):
            all_res.update(pickle.loads(recv[: shape[0]].cpu().numpy().tobytes()))
        return all_res
