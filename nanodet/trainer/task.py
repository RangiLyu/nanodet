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

import copy
import os
import warnings
import json
import torch
import logging
from pytorch_lightning import LightningModule
from typing import Any, List, Dict, Tuple, Optional

from ..model.arch import build_model
from nanodet.util import mkdir


class TrainingTask(LightningModule):
    """
    Pytorch Lightning module of a general training task.
    Including training, evaluating and testing.
    Args:
        cfg: Training configurations
        evaluator: Evaluator for evaluating the model performance.
    """

    def __init__(self, cfg, evaluator=None):
        super(TrainingTask, self).__init__()
        self.cfg = cfg
        self.model = build_model(cfg.model)
        self.evaluator = evaluator
        self.save_flag = -10
        self.log_style = 'NanoDet'  # Log style. Choose between 'NanoDet' or 'Lightning'
        # TODO: use callback to log

    def forward(self, x):
        x = self.model(x)
        return x

    @torch.no_grad()
    def predict(self, batch, batch_idx=None, dataloader_idx=None):
        preds = self.forward(batch['img'])
        results = self.model.head.post_process(preds, batch)
        return results

    def on_train_start(self) -> None:
        self.lr_scheduler.last_epoch = self.current_epoch-1

    def training_step(self, batch, batch_idx):
        preds, loss, loss_states = self.model.forward_train(batch)

        # log train losses
        if self.log_style == 'Lightning':
            self.log('lr', self.optimizers().param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=True)
            for k, v in loss_states.items():
                self.log('Train/'+k, v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        elif self.log_style == 'NanoDet' and self.global_step % self.cfg.log.interval == 0:
            lr = self.optimizers().param_groups[0]['lr']
            log_msg = 'Train|Epoch{}/{}|Iter{}({})| lr:{:.2e}| '.format(self.current_epoch+1,
                self.cfg.schedule.total_epochs, self.global_step, batch_idx, lr)
            self.scalar_summary('Train_loss/lr', 'Train', lr, self.global_step)
            for l in loss_states:
                log_msg += '{}:{:.4f}| '.format(l, loss_states[l].mean().item())
                self.scalar_summary('Train_loss/' + l, 'Train', loss_states[l].mean().item(), self.global_step)
            self.info(log_msg)

        return loss

    def training_epoch_end(self, outputs: List[Any]) -> None:
        self.trainer.save_checkpoint(os.path.join(self.cfg.save_dir, 'model_last.ckpt'))
        self.lr_scheduler.step()

    def validation_step(self, batch, batch_idx):
        preds, loss, loss_states = self.model.forward_train(batch)

        if self.log_style == 'Lightning':
            self.log('Val/loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=False)
            for k, v in loss_states.items():
                self.log('Val/' + k, v, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        elif self.log_style == 'NanoDet' and batch_idx % self.cfg.log.interval == 0:
            lr = self.optimizers().param_groups[0]['lr']
            log_msg = 'Val|Epoch{}/{}|Iter{}({})| lr:{:.2e}| '.format(self.current_epoch+1,
                self.cfg.schedule.total_epochs, self.global_step, batch_idx, lr)
            for l in loss_states:
                log_msg += '{}:{:.4f}| '.format(l, loss_states[l].mean().item())
            self.info(log_msg)

        dets = self.model.head.post_process(preds, batch)
        return dets

    def validation_epoch_end(self, validation_step_outputs):
        """
        Called at the end of the validation epoch with the outputs of all validation steps.
        Evaluating results and save best model.
        Args:
            validation_step_outputs: A list of val outputs

        """
        results = {}
        for res in validation_step_outputs:
            results.update(res)
        eval_results = self.evaluator.evaluate(results, self.cfg.save_dir, rank=self.local_rank)
        metric = eval_results[self.cfg.evaluator.save_key]
        # save best model
        if metric > self.save_flag:
            self.save_flag = metric
            best_save_path = os.path.join(self.cfg.save_dir, 'model_best')
            mkdir(self.local_rank, best_save_path)
            self.trainer.save_checkpoint(os.path.join(best_save_path, "model_best.ckpt"))
            txt_path = os.path.join(best_save_path, "eval_results.txt")
            if self.local_rank < 1:
                with open(txt_path, "a") as f:
                    f.write("Epoch:{}\n".format(self.current_epoch+1))
                    for k, v in eval_results.items():
                        f.write("{}: {}\n".format(k, v))
        else:
            warnings.warn('Warning! Save_key is not in eval results! Only save model last!')
        if self.log_style == 'Lightning':
            for k, v in eval_results.items():
                self.log('Val_metrics/' + k, v, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        elif self.log_style == 'NanoDet':
            for k, v in eval_results.items():
                self.scalar_summary('Val_metrics/' + k, 'Val', v, self.current_epoch+1)

    def test_step(self, batch, batch_idx):
        dets = self.predict(batch, batch_idx)
        return dets

    def test_epoch_end(self, test_step_outputs):
        results = {}
        for res in test_step_outputs:
            results.update(res)
        res_json = self.evaluator.results2json(results)
        json_path = os.path.join(self.cfg.save_dir, 'results.json')
        json.dump(res_json, open(json_path, 'w'))

        if self.cfg.test_mode == 'val':
            eval_results = self.evaluator.evaluate(results, self.cfg.save_dir, rank=self.local_rank)
            txt_path = os.path.join(self.cfg.save_dir, "eval_results.txt")
            with open(txt_path, "a") as f:
                for k, v in eval_results.items():
                    f.write("{}: {}\n".format(k, v))

    def configure_optimizers(self):
        """
        Prepare optimizer and learning-rate scheduler
        to use in optimization.

        Returns:
            optimizer
        """
        optimizer_cfg = copy.deepcopy(self.cfg.schedule.optimizer)
        name = optimizer_cfg.pop('name')
        build_optimizer = getattr(torch.optim, name)
        optimizer = build_optimizer(params=self.parameters(), **optimizer_cfg)

        schedule_cfg = copy.deepcopy(self.cfg.schedule.lr_schedule)
        name = schedule_cfg.pop('name')
        build_scheduler = getattr(torch.optim.lr_scheduler, name)
        self.lr_scheduler = build_scheduler(optimizer=optimizer, **schedule_cfg)
        # lr_scheduler = {'scheduler': self.lr_scheduler,
        #                 'interval': 'epoch',
        #                 'frequency': 1}
        # return [optimizer], [lr_scheduler]

        return optimizer

    def optimizer_step(self,
                       epoch=None,
                       batch_idx=None,
                       optimizer=None,
                       optimizer_idx=None,
                       optimizer_closure=None,
                       on_tpu=None,
                       using_native_amp=None,
                       using_lbfgs=None):
        """
        Performs a single optimization step (parameter update).
        Args:
            epoch: Current epoch
            batch_idx: Index of current batch
            optimizer: A PyTorch optimizer
            optimizer_idx: If you used multiple optimizers this indexes into that list.
            optimizer_closure: closure for all optimizers
            on_tpu: true if TPU backward is required
            using_native_amp: True if using native amp
            using_lbfgs: True if the matching optimizer is lbfgs
        """
        # warm up lr
        if self.trainer.global_step <= self.cfg.schedule.warmup.steps:
            if self.cfg.schedule.warmup.name == 'constant':
                warmup_lr = self.cfg.schedule.optimizer.lr * self.cfg.schedule.warmup.ratio
            elif self.cfg.schedule.warmup.name == 'linear':
                k = (1 - self.trainer.global_step / self.cfg.schedule.warmup.steps) * (1 - self.cfg.schedule.warmup.ratio)
                warmup_lr = self.cfg.schedule.optimizer.lr * (1 - k)
            elif self.cfg.schedule.warmup.name == 'exp':
                k = self.cfg.schedule.warmup.ratio ** (1 - self.trainer.global_step / self.cfg.schedule.warmup.steps)
                warmup_lr = self.cfg.schedule.optimizer.lr * k
            else:
                raise Exception('Unsupported warm up type!')
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        items.pop("loss", None)
        return items

    def scalar_summary(self, tag, phase, value, step):
        """
        Write Tensorboard scalar summary log.
        Args:
            tag: Name for the tag
            phase: 'Train' or 'Val'
            value: Value to record
            step: Step value to record

        """
        if self.local_rank < 1:
            self.logger.experiment.add_scalars(tag, {phase: value}, step)

    def info(self, string):
        if self.local_rank < 1:
            logging.info(string)








