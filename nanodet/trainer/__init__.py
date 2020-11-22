import torch
from .trainer import Trainer
from .dist_trainer import DistTrainer


def build_trainer(rank, cfg, model, logger):
    if len(cfg.device.gpu_ids) > 1:
        trainer = DistTrainer(rank, cfg, model, logger)
        trainer.set_device(cfg.device.batchsize_per_gpu, rank, device=torch.device('cuda'))  # TODO: device
    else:
        trainer = Trainer(rank, cfg, model, logger)
        trainer.set_device(cfg.device.batchsize_per_gpu, cfg.device.gpu_ids, device=torch.device('cuda'))
    return trainer

