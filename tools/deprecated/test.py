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

import argparse
import datetime
import json
import os
import warnings

import torch

from nanodet.data.collate import collate_function
from nanodet.data.dataset import build_dataset
from nanodet.evaluator import build_evaluator
from nanodet.model.arch import build_model
from nanodet.trainer import build_trainer
from nanodet.util import Logger, cfg, load_config, mkdir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", type=str, default="val", help="task to run, test or val"
    )
    parser.add_argument("--config", type=str, help="model config file(.yml) path")
    parser.add_argument("--model", type=str, help="model weight file(.pth) path")
    parser.add_argument(
        "--save_result",
        action="store_true",
        default=True,
        help="save val results to txt",
    )
    args = parser.parse_args()
    return args


def main(args):
    warnings.warn(
        "Warning! Old testing code is deprecated and will be deleted "
        "in next version. Please use tools/test.py"
    )
    load_config(cfg, args.config)
    local_rank = -1
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    cfg.defrost()
    timestr = datetime.datetime.now().__format__("%Y%m%d%H%M%S")
    cfg.save_dir = os.path.join(cfg.save_dir, timestr)
    cfg.freeze()
    mkdir(local_rank, cfg.save_dir)
    logger = Logger(local_rank, cfg.save_dir)

    logger.log("Creating model...")
    model = build_model(cfg.model)

    logger.log("Setting up data...")
    val_dataset = build_dataset(cfg.data.val, args.task)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=False,
        num_workers=cfg.device.workers_per_gpu,
        pin_memory=True,
        collate_fn=collate_function,
        drop_last=True,
    )
    trainer = build_trainer(local_rank, cfg, model, logger)
    cfg.schedule.update({"load_model": args.model})
    trainer.load_model(cfg)
    evaluator = build_evaluator(cfg.evaluator, val_dataset)
    logger.log("Starting testing...")
    with torch.no_grad():
        results, val_loss_dict = trainer.run_epoch(0, val_dataloader, mode=args.task)
    if args.task == "test":
        res_json = evaluator.results2json(results)
        json_path = os.path.join(cfg.save_dir, "results{}.json".format(timestr))
        json.dump(res_json, open(json_path, "w"))
    elif args.task == "val":
        eval_results = evaluator.evaluate(results, cfg.save_dir, rank=local_rank)
        if args.save_result:
            txt_path = os.path.join(cfg.save_dir, "eval_results{}.txt".format(timestr))
            with open(txt_path, "a") as f:
                for k, v in eval_results.items():
                    f.write("{}: {}\n".format(k, v))


if __name__ == "__main__":
    args = parse_args()
    main(args)
