import copy
from .fpn import FPN
from .pan import PAN


def build_fpn(cfg):
    fpn_cfg = copy.deepcopy(cfg)
    name = fpn_cfg.pop('name')
    if name == 'FPN':
        return FPN(**fpn_cfg)
    elif name == 'PAN':
        return PAN(**fpn_cfg)
    else:
        raise NotImplementedError