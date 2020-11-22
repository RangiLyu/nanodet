from .gfl import GFL


def build_model(model_cfg):
    if model_cfg.arch.name == 'GFL':
        model = GFL(model_cfg.arch.backbone, model_cfg.arch.fpn, model_cfg.arch.head)
    else:
        raise NotImplementedError
    return model
