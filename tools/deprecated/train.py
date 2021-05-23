import os
import torch
import logging
import warnings
import argparse
import numpy as np
import torch.distributed as dist

from nanodet.util import mkdir, Logger, cfg, load_config
from nanodet.trainer import build_trainer
from nanodet.data.collate import collate_function
from nanodet.data.dataset import build_dataset
from nanodet.model.arch import build_model
from nanodet.evaluator import build_evaluator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed')
    args = parser.parse_args()
    return args


def init_seeds(seed=0):
    """
    manually set a random seed for numpy, torch and cuda
    :param seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(args):
    warnings.warn('Warning! Old training code is deprecated and will be deleted '
                  'in next version. Please use tools/train.py')
    load_config(cfg, args.config)
    local_rank = int(args.local_rank)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    mkdir(local_rank, cfg.save_dir)
    logger = Logger(local_rank, cfg.save_dir)
    if args.seed is not None:
        logger.log('Set random seed to {}'.format(args.seed))
        init_seeds(args.seed)

    logger.log('Creating model...')
    model = build_model(cfg.model)

    logger.log('Setting up data...')
    train_dataset = build_dataset(cfg.data.train, 'train')
    val_dataset = build_dataset(cfg.data.val, 'test')

    if len(cfg.device.gpu_ids) > 1:
        print('rank = ', local_rank)
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(local_rank % num_gpus)
        dist.init_process_group(backend='nccl')
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.device.batchsize_per_gpu,
                                                       num_workers=cfg.device.workers_per_gpu, pin_memory=True,
                                                       collate_fn=collate_function, sampler=train_sampler,
                                                       drop_last=True)
    else:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.device.batchsize_per_gpu,
                                                       shuffle=True, num_workers=cfg.device.workers_per_gpu,
                                                       pin_memory=True, collate_fn=collate_function, drop_last=True)

    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.device.batchsize_per_gpu,
                                                 shuffle=False, num_workers=cfg.device.workers_per_gpu,
                                                 pin_memory=True, collate_fn=collate_function, drop_last=True)

    trainer = build_trainer(local_rank, cfg, model, logger)

    if 'load_model' in cfg.schedule:
        trainer.load_model(cfg)
    if 'resume' in cfg.schedule:
        trainer.resume(cfg)

    evaluator = build_evaluator(cfg, val_dataset)

    logger.log('Starting training...')
    trainer.run(train_dataloader, val_dataloader, evaluator)


if __name__ == '__main__':
    args = parse_args()
    main(args)
