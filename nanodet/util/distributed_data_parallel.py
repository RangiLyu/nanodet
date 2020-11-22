from torch.nn.parallel import DistributedDataParallel
from .scatter_gather import scatter_kwargs


class DDP(DistributedDataParallel):

    def __init__(self, batchsize, **kwargs):
        self.batchsize = batchsize
        super(DDP, self).__init__(**kwargs)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim, chunk_sizes=[self.batchsize])