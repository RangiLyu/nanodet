# Copyright 2021 RangiLyu. All rights reserved.
# =====================================================================
# Modified from: https://github.com/facebookresearch/d2go
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Licensed under the Apache License, Version 2.0 (the "License")
import itertools
import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class ExpMovingAverager(object):
    """Exponential Moving Average.

    Args:
        decay (float): EMA decay factor, should be in [0, 1]. A decay of 0 corresponds
            to always using the latest value (no EMA) and a decay of 1 corresponds to
            not updating weights after initialization. Default to 0.9998.
        device (str): If not None, move EMA state to device.
    """

    def __init__(self, decay: float = 0.9998, device: Optional[str] = None):
        if decay < 0 or decay > 1.0:
            raise ValueError(f"Decay should be in [0, 1], {decay} was given.")
        self.decay: float = decay
        self.state: Dict[str, Any] = {}
        self.device: Optional[str] = device

    def load_from(self, model: nn.Module) -> None:
        """Load state from the model."""
        self.state.clear()
        for name, val in self._get_model_state_iterator(model):
            val = val.detach().clone()
            self.state[name] = val.to(self.device) if self.device else val

    def has_inited(self) -> bool:
        return len(self.state) > 0

    def apply_to(self, model: nn.Module) -> None:
        """Apply EMA state to the model."""
        with torch.no_grad():
            for name, val in self._get_model_state_iterator(model):
                assert (
                    name in self.state
                ), f"Name {name} not exist, available names are {self.state.keys()}"
                val.copy_(self.state[name])

    def state_dict(self) -> Dict[str, Any]:
        return self.state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.state.clear()
        for name, val in state_dict.items():
            self.state[name] = val.to(self.device) if self.device else val

    def to(self, device: torch.device) -> None:
        """moves EMA state to device."""
        for name, val in self.state.items():
            self.state[name] = val.to(device)

    def _get_model_state_iterator(self, model: nn.Module):
        param_iter = model.named_parameters()
        # pyre-fixme[16]: `nn.Module` has no attribute `named_buffers`.
        buffer_iter = model.named_buffers()
        return itertools.chain(param_iter, buffer_iter)

    def calculate_dacay(self, iteration: int) -> float:
        decay = (self.decay) * math.exp(-(1 + iteration) / 2000) + (1 - self.decay)
        return decay

    def update(self, model: nn.Module, iteration: int) -> None:
        decay = self.calculate_dacay(iteration)
        with torch.no_grad():
            for name, val in self._get_model_state_iterator(model):
                ema_val = self.state[name]
                if self.device:
                    val = val.to(self.device)
                ema_val.copy_(ema_val * (1 - decay) + val * decay)
