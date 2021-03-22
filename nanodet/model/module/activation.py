import torch.nn as nn

activations = {'ReLU': nn.ReLU,
               'LeakyReLU': nn.LeakyReLU,
               'ReLU6': nn.ReLU6,
               'SELU': nn.SELU,
               'ELU': nn.ELU,
               'GELU': nn.GELU,
               None: nn.Identity
               }


def act_layers(name):
    assert name in activations.keys()
    if name == 'LeakyReLU':
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif name == 'GELU':
        return nn.GELU()
    else:
        return activations[name](inplace=True)
