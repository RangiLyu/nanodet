import copy
import logging

import torch
from torch.nn import GroupNorm, LayerNorm
from torch.nn.modules.batchnorm import _BatchNorm

NORMS = (GroupNorm, LayerNorm, _BatchNorm)


def build_optimizer(model, config):
    """Build optimizer from config.

    Supports customised parameter-level hyperparameters.
    The config should be like:
    >>> optimizer:
    >>>   name: AdamW
    >>>   lr: 0.001
    >>>   weight_decay: 0.05
    >>>   no_norm_decay: True
    >>>   param_level_cfg:  # parameter-level config
    >>>     backbone:
    >>>       lr_mult: 0.1
    """
    config = copy.deepcopy(config)
    param_dict = {}
    no_norm_decay = config.pop("no_norm_decay", False)
    no_bias_decay = config.pop("no_bias_decay", False)
    param_level_cfg = config.pop("param_level_cfg", {})
    base_lr = config.get("lr", None)
    base_wd = config.get("weight_decay", None)

    name = config.pop("name")
    optim_cls = getattr(torch.optim, name)

    logger = logging.getLogger("NanoDet")

    # custom param-wise lr and weight_decay
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        param_dict[p] = {"name": name}

        for key in param_level_cfg:
            if key in name:
                if "lr_mult" in param_level_cfg[key] and base_lr:
                    param_dict[p].update(
                        {"lr": base_lr * param_level_cfg[key]["lr_mult"]}
                    )
                if "decay_mult" in param_level_cfg[key] and base_wd:
                    param_dict[p].update(
                        {"weight_decay": base_wd * param_level_cfg[key]["decay_mult"]}
                    )
                break
    if no_norm_decay:
        # update norms decay
        for name, m in model.named_modules():
            if isinstance(m, NORMS):
                param_dict[m.bias].update({"weight_decay": 0})
                param_dict[m.weight].update({"weight_decay": 0})
    if no_bias_decay:
        # update bias decay
        for name, m in model.named_modules():
            if hasattr(m, "bias"):
                param_dict[m.bias].update({"weight_decay": 0})

    # convert param dict to optimizer's param groups
    param_groups = []
    for p, pconfig in param_dict.items():
        name = pconfig.pop("name", None)
        if "weight_decay" in pconfig or "lr" in pconfig:
            logger.info(f"special optimizer hyperparameter: {name} - {pconfig}")
        param_groups += [{"params": p, **pconfig}]

    optimizer = optim_cls(param_groups, **config)
    return optimizer
