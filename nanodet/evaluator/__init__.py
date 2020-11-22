from .coco_detection import CocoDetectionEvaluator


def build_evaluator(cfg, dataset):
    if cfg.evaluator.name == 'CocoDetectionEvaluator':
        return CocoDetectionEvaluator(dataset)
    else:
        raise NotImplementedError
