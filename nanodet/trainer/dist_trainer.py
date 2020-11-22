import torch.distributed as dist
from .trainer import Trainer
from ..util import DDP


def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size



class DistTrainer(Trainer):
    """
    Distributed trainer for multi-gpu training. (not finish yet)
    """
    def run_step(self, model, batch, mode='train'):
        output, loss, loss_stats = model.module.forward_train(batch)
        loss = loss.mean()
        if mode == 'train':
            self.optimizer.zero_grad()
            loss.backward()
            average_gradients(model)
            self.optimizer.step()
        return output, loss, loss_stats

    def set_device(self, batch_per_gpu, rank, device):
        """
        Set model device for Distributed-Data-Parallel
        :param batch_per_gpu: batch size of each gpu
        :param rank: distributed training process rank
        :param device: cuda
        """
        self.rank = rank
        self.model = DDP(batch_per_gpu, module=self.model.cuda(), device_ids=[rank], output_device=rank)


