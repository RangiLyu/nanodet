import copy
from .gfl_head import GFLHead
from .nanodet_head import NanoDetHead


def build_head(cfg):
    head_cfg = copy.deepcopy(cfg)
    name = head_cfg.pop('name')
    if name == 'GFLHead':
        return GFLHead(**head_cfg)
    elif name == 'NanoDetHead':
        return NanoDetHead(**head_cfg)
    else:
        raise NotImplementedError