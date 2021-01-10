import torch
import torch.nn as nn
from ..module.conv import ConvModule


class TinyResBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, norm_cfg, activation, res_type='concat'):
        super(TinyResBlock, self).__init__()
        assert in_channels % 2 == 0
        assert res_type in ['concat', 'add']
        self.res_type = res_type
        self.in_conv = ConvModule(in_channels, in_channels//2, kernel_size, padding=(kernel_size-1)//2, norm_cfg=norm_cfg, activation=activation)
        self.mid_conv = ConvModule(in_channels//2, in_channels//2, kernel_size, padding=(kernel_size-1)//2, norm_cfg=norm_cfg, activation=activation)
        if res_type == 'add':
            self.out_conv = ConvModule(in_channels//2, in_channels, kernel_size, padding=(kernel_size-1)//2, norm_cfg=norm_cfg, activation=activation)

    def forward(self, x):
        x = self.in_conv(x)
        x1 = self.mid_conv(x)
        if self.res_type == 'add':
            return self.out_conv(x+x1)
        else:
            return torch.cat((x1, x), dim=1)


class CspBlock(nn.Module):
    def __init__(self, in_channels, num_res, kernel_size=3, stride=0, norm_cfg=dict(type='BN', requires_grad=True), activation='LeakyReLU'):
        super(CspBlock, self).__init__()
        assert in_channels % 2 == 0
        self.in_conv = ConvModule(in_channels, in_channels, kernel_size, stride, padding=(kernel_size-1)//2, norm_cfg=norm_cfg, activation=activation)
        res_blocks = []
        for i in range(num_res):
            res_block = TinyResBlock(in_channels, kernel_size, norm_cfg, activation)
            res_blocks.append(res_block)
        self.res_blocks = nn.Sequential(*res_blocks)
        self.res_out_conv = ConvModule(in_channels, in_channels, kernel_size, padding=(kernel_size-1)//2, norm_cfg=norm_cfg, activation=activation)

    def forward(self, x):
        x = self.in_conv(x)
        x1 = self.res_blocks(x)
        x1 = self.res_out_conv(x1)
        out = torch.cat((x1, x), dim=1)
        return out


class CustomCspNet(nn.Module):
    def __init__(self, net_cfg, out_stages, norm_cfg=dict(type='BN', requires_grad=True), activation='LeakyReLU'):
        super(CustomCspNet, self).__init__()
        self.out_stages = out_stages
        self.activation = activation
        self.stages = nn.ModuleList()
        for stage_cfg in net_cfg:
            if stage_cfg[0] == 'Conv':
                in_channels, out_channels, kernel_size, stride = stage_cfg[1:]
                stage = ConvModule(in_channels, out_channels, kernel_size, stride, padding=(kernel_size-1)//2, norm_cfg=norm_cfg, activation=activation)
            elif stage_cfg[0] == 'CspBlock':
                in_channels, num_res, kernel_size, stride = stage_cfg[1:]
                stage = CspBlock(in_channels, num_res, kernel_size, stride, norm_cfg, activation)
            elif stage_cfg[0] == 'MaxPool':
                kernel_size, stride = stage_cfg[1:]
                stage = nn.MaxPool2d(kernel_size, stride, padding=(kernel_size-1)//2)
            else:
                raise ModuleNotFoundError
            self.stages.append(stage)
        self._init_weight()
    
    def forward(self, x):
        output = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_stages:
                output.append(x)
        return tuple(output)

    def _init_weight(self):
        for m in self.modules():
            if self.activation == 'LeakyReLU':
                nonlinearity = 'leaky_relu'
            else:
                nonlinearity = 'relu'
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=nonlinearity)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


