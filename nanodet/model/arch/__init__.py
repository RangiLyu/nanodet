import warnings

from .one_stage_detector import OneStageDetector


def build_model(model_cfg):
    if model_cfg.arch.name == 'GFL':
        warnings.warn("Model architecture name is changed to 'OneStageDetector'. "
                      "The name 'GFL' is deprecated, please change the model->arch->name "
                      "in your YAML config file to OneStageDetector. ")
        model = OneStageDetector(model_cfg.arch.backbone, model_cfg.arch.fpn, model_cfg.arch.head)
    elif model_cfg.arch.name == 'OneStageDetector':
        model = OneStageDetector(model_cfg.arch.backbone, model_cfg.arch.fpn, model_cfg.arch.head)
    else:
        raise NotImplementedError
    return model
