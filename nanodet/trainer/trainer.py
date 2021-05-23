import os
import copy
import warnings
import torch
from nanodet.util import mkdir, DataParallel, load_model_weight, save_model, MovingAverage, AverageMeter


class Trainer:
    """
    Epoch based trainer
    """
    def __init__(self, rank, cfg, model, logger):
        self.rank = rank  # local rank for distributed training. For single gpu training, default is -1
        self.cfg = cfg
        self.model = model
        self.logger = logger
        self._init_optimizer()
        self._iter = 1
        self.epoch = 1

    def set_device(self, batch_per_gpu, gpu_ids, device):
        """
        Set model device to GPU.
        :param batch_per_gpu: batch size of each gpu
        :param gpu_ids: a list of gpu ids
        :param device: cuda
        """
        num_gpu = len(gpu_ids)
        batch_sizes = [batch_per_gpu for i in range(num_gpu)]
        self.logger.log('Training batch size: {}'.format(batch_per_gpu*num_gpu))
        self.model = DataParallel(self.model, gpu_ids, chunk_sizes=batch_sizes).to(device)

    def _init_optimizer(self):
        optimizer_cfg = copy.deepcopy(self.cfg.schedule.optimizer)
        name = optimizer_cfg.pop('name')
        Optimizer = getattr(torch.optim, name)
        self.optimizer = Optimizer(params=self.model.parameters(), **optimizer_cfg)

    def _init_scheduler(self):
        schedule_cfg = copy.deepcopy(self.cfg.schedule.lr_schedule)
        name = schedule_cfg.pop('name')
        Scheduler = getattr(torch.optim.lr_scheduler, name)
        self.lr_scheduler = Scheduler(optimizer=self.optimizer, **schedule_cfg)

    def run_step(self, model, meta, mode='train'):
        """
        Training step including forward and backward
        :param model: model to train
        :param meta: a batch of input data
        :param mode: train or val or test
        :return: result, total loss and a dict of all losses
        """
        output, loss, loss_dict = model.module.forward_train(meta)
        loss = loss.mean()
        if mode == 'train':
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return output, loss, loss_dict

    def run_epoch(self, epoch, data_loader, mode):
        """
        train or validate one epoch
        :param epoch: current epoch number
        :param data_loader: dataloader of train or test dataset
        :param mode: train or val or test
        :return: outputs and a dict of epoch average losses
        """
        model = self.model
        if mode == 'train':
            model.train()
            if self.rank > -1:  # Using distributed training, need to set epoch for sampler
                self.logger.log("distributed sampler set epoch at {}".format(epoch))
                data_loader.sampler.set_epoch(epoch)
        else:
            model.eval()
            torch.cuda.empty_cache()
        results = {}
        epoch_losses = {}
        step_losses = {}
        num_iters = len(data_loader)
        for iter_id, meta in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            meta['img'] = meta['img'].to(device=torch.device('cuda'), non_blocking=True)
            output, loss, loss_stats = self.run_step(model, meta, mode)
            if mode == 'val' or mode == 'test':
                batch_dets = model.module.head.post_process(output, meta)
                results.update(batch_dets)
            for k in loss_stats:
                if k not in epoch_losses:
                    epoch_losses[k] = AverageMeter(loss_stats[k].mean().item())
                    step_losses[k] = MovingAverage(loss_stats[k].mean().item(), window_size=self.cfg.log.interval)
                else:
                    epoch_losses[k].update(loss_stats[k].mean().item())
                    step_losses[k].push(loss_stats[k].mean().item())

            if iter_id % self.cfg.log.interval == 0:
                log_msg = '{}|Epoch{}/{}|Iter{}({}/{})| lr:{:.2e}| '.format(mode, epoch, self.cfg.schedule.total_epochs,
                    self._iter, iter_id, num_iters, self.optimizer.param_groups[0]['lr'])
                for l in step_losses:
                    log_msg += '{}:{:.4f}| '.format(l, step_losses[l].avg())
                    if mode == 'train' and self.rank < 1:
                        self.logger.scalar_summary('Train_loss/' + l, mode, step_losses[l].avg(), self._iter)
                self.logger.log(log_msg)
            if mode == 'train':
                self._iter += 1
            del output, loss, loss_stats
        epoch_loss_dict = {k: v.avg for k, v in epoch_losses.items()}
        return results, epoch_loss_dict

    def run(self, train_loader, val_loader, evaluator):
        """
        start running
        :param train_loader:
        :param val_loader:
        :param evaluator:
        """
        start_epoch = self.epoch
        save_flag = -10
        if self.cfg.schedule.warmup.steps > 0 and start_epoch == 1:
            self.logger.log('Start warming up...')
            self.warm_up(train_loader)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.cfg.schedule.optimizer.lr

        self._init_scheduler()
        self.lr_scheduler.last_epoch = start_epoch - 1
        
        # resume learning rate of last epoch
        if start_epoch > 1:
            for param_group, lr in zip(self.optimizer.param_groups, self.lr_scheduler.get_lr()):
                param_group['lr'] = lr

        for epoch in range(start_epoch, self.cfg.schedule.total_epochs + 1):
            results, train_loss_dict = self.run_epoch(epoch, train_loader, mode='train')
            self.lr_scheduler.step()
            save_model(self.rank, self.model, os.path.join(self.cfg.save_dir, 'model_last.pth'), epoch, self._iter, self.optimizer)
            for k, v in train_loss_dict.items():
                self.logger.scalar_summary('Epoch_loss/' + k, 'train', v, epoch)

            # --------evaluate----------
            if self.cfg.schedule.val_intervals > 0 and epoch % self.cfg.schedule.val_intervals == 0:
                with torch.no_grad():
                    results, val_loss_dict = self.run_epoch(self.epoch, val_loader, mode='val')
                for k, v in val_loss_dict.items():
                    self.logger.scalar_summary('Epoch_loss/' + k, 'val', v, epoch)
                eval_results = evaluator.evaluate(results, self.cfg.save_dir, rank=self.rank)
                for k, v in eval_results.items():
                    self.logger.scalar_summary('Val_metrics/' + k, 'val', v, epoch)
                if self.cfg.evaluator.save_key in eval_results:
                    metric = eval_results[self.cfg.evaluator.save_key]
                    if metric > save_flag:
                        # ------save best model--------
                        save_flag = metric
                        best_save_path = os.path.join(self.cfg.save_dir, 'model_best')
                        mkdir(self.rank, best_save_path)
                        save_model(self.rank, self.model, os.path.join(best_save_path, 'model_best.pth'), epoch,
                                   self._iter, self.optimizer)
                        txt_path = os.path.join(best_save_path, "eval_results.txt")
                        if self.rank < 1:
                            with open(txt_path, "a") as f:
                                f.write("Epoch:{}\n".format(epoch))
                                for k, v in eval_results.items():
                                    f.write("{}: {}\n".format(k, v))
                else:
                    warnings.warn('Warning! Save_key is not in eval results! Only save model last!')
            self.epoch += 1

    def get_warmup_lr(self, cur_iters):
        if self.cfg.schedule.warmup.name == 'constant':
            warmup_lr = self.cfg.schedule.optimizer.lr * self.cfg.schedule.warmup.ratio
        elif self.cfg.schedule.warmup.name == 'linear':
            k = (1 - cur_iters / self.cfg.schedule.warmup.steps) * (1 - self.cfg.schedule.warmup.ratio)
            warmup_lr = self.cfg.schedule.optimizer.lr * (1 - k)
        elif self.cfg.schedule.warmup.name == 'exp':
            k = self.cfg.schedule.warmup.ratio ** (1 - cur_iters / self.cfg.schedule.warmup.steps)
            warmup_lr = self.cfg.schedule.optimizer.lr * k
        else:
            raise Exception('Unsupported warm up type!')
        return warmup_lr

    def warm_up(self, data_loader):
        model = self.model
        model.train()
        step_losses = {}
        num_iters = self.cfg.schedule.warmup.steps
        cur_iter = 0
        while cur_iter < num_iters:
            for iter_id, batch in enumerate(data_loader):
                cur_iter += 1
                if cur_iter >= num_iters:
                    break
                lr = self.get_warmup_lr(cur_iter)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                batch['img'] = batch['img'].to(device=torch.device('cuda'), non_blocking=True)
                output, loss, loss_stats = self.run_step(model, batch)

                # TODO: simplify code
                for k in loss_stats:
                    if k not in step_losses:
                        step_losses[k] = MovingAverage(loss_stats[k].mean().item(), window_size=self.cfg.log.interval)
                    else:
                        step_losses[k].push(loss_stats[k].mean().item())
                if iter_id % self.cfg.log.interval == 0:
                    log_msg = '{}|Iter({}/{})| lr:{:.2e}| '.format('warmup', cur_iter, num_iters, self.optimizer.param_groups[0]['lr'])
                    for l in step_losses:
                        log_msg += '{}:{:.4f}| '.format(l, step_losses[l].avg())
                    self.logger.log(log_msg)
                del output, loss, loss_stats

    def load_model(self, cfg):
        load_path = cfg.schedule.load_model
        checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
        self.logger.log('loaded {}, epoch {}'.format(load_path, checkpoint['epoch']))
        if hasattr(self.model, 'module'):
            load_model_weight(self.model.module, checkpoint, self.logger)
        else:
            load_model_weight(self.model, checkpoint, self.logger)

    def resume(self, cfg):
        """
        load model and optimizer state
        """
        if cfg.schedule.resume is not None:
            load_path = cfg.schedule.resume
        else:
            load_path = os.path.join(cfg.save_dir, 'model_last.pth')
        checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
        self.logger.log('loaded {}, epoch {}'.format(load_path, checkpoint['epoch']))
        if hasattr(self.model, 'module'):
            load_model_weight(self.model.module, checkpoint, self.logger)
        else:
            load_model_weight(self.model, checkpoint, self.logger)
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epoch = checkpoint['epoch'] + 1
            self.logger.log('resumed at epoch: {}'.format(self.epoch))
            if 'iter' in checkpoint:
                self._iter = checkpoint['iter'] + 1
                self.logger.log('resumed at steps: {}'.format(self._iter))
        else:
            self.logger.log('No optimizer parameters in checkpoint.')

