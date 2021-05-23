# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import json
import datetime
import argparse
import warnings
import pytorch_lightning as pl

from nanodet.util import mkdir, Logger, cfg, load_config, convert_old_model
from nanodet.data.collate import collate_function
from nanodet.data.dataset import build_dataset
from nanodet.trainer.task import TrainingTask
from nanodet.evaluator import build_evaluator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='val', help='task to run, test or val')
    parser.add_argument('--config', type=str, help='model config file(.yml) path')
    parser.add_argument('--model', type=str, help='ckeckpoint file(.ckpt) path')
    args = parser.parse_args()
    return args


def main(args):
    load_config(cfg, args.config)
    local_rank = -1
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    cfg.defrost()
    timestr = datetime.datetime.now().__format__('%Y%m%d%H%M%S')
    cfg.save_dir = os.path.join(cfg.save_dir, timestr)
    mkdir(local_rank, cfg.save_dir)
    logger = Logger(local_rank, cfg.save_dir)

    assert args.task in ['val', 'test']
    cfg.update({'test_mode': args.task})

    logger.log('Setting up data...')
    val_dataset = build_dataset(cfg.data.val, args.task)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.device.batchsize_per_gpu,
                                                 shuffle=False, num_workers=cfg.device.workers_per_gpu,
                                                 pin_memory=True, collate_fn=collate_function, drop_last=True)
    evaluator = build_evaluator(cfg, val_dataset)

    logger.log('Creating model...')
    task = TrainingTask(cfg, evaluator)

    ckpt = torch.load(args.model)
    if 'pytorch-lightning_version' not in ckpt:
        warnings.warn('Warning! Old .pth checkpoint is deprecated. '
                      'Convert the checkpoint with tools/convert_old_checkpoint.py ')
        ckpt = convert_old_model(ckpt)
    task.load_state_dict(ckpt['state_dict'])

    trainer = pl.Trainer(default_root_dir=cfg.save_dir,
                         gpus=cfg.device.gpu_ids,
                         accelerator='ddp',
                         log_every_n_steps=cfg.log.interval,
                         num_sanity_val_steps=0,
                         )
    logger.log('Starting testing...')
    trainer.test(task, val_dataloader)


if __name__ == '__main__':
    args = parse_args()
    main(args)
