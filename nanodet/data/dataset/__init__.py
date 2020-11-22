import copy
from .coco import CocoDataset


def build_dataset(cfg, mode):
    dataset_cfg = copy.deepcopy(cfg)
    if dataset_cfg['name'] == 'coco':
        dataset_cfg.pop('name')
        return CocoDataset(mode=mode, **dataset_cfg)
