# -*- coding: utf-8 -*-
import time
import torch
from torch import nn


class Net_Builder(torch.nn.Module):
    def __init__(self, inner_shape, net_stracture, out_shape):
        super(Net_Builder, self).__init__()
        self.inner_shape = inner_shape
        self.out_shape = out_shape
        self.net_stracture = net_stracture
        for idx, hidden_unit in enumerate(self.net_stracture):

            if not idx:
                exec(f'self.dense{idx} = nn.Linear(self.inner_shape, {hidden_unit})')
            else:
                exec(f'self.dense{idx} = nn.Linear({self.net_stracture[idx-1]}, {hidden_unit})')

            exec(f'self.act{idx} = nn.ReLU(inplace=True)')

    def forward(self, input_tensor):
        # 在局部域中使用exec变量消失解决方法，调用exec之前通过locals()获得局部字典，返回值从locals中找
        loc = locals()
        for idx, _ in enumerate(self.net_stracture):
            if not idx:
                exec(f"common = self.dense{idx}(input_tensor)")
                exec(f"common = self.act{idx}(common)")
            else:
                exec(f"common = self.dense{idx}(common)")
                exec(f"common = self.act{idx}(common)")
        return loc['common']


class Model(torch.nn.Module):
    def __init__(self, inner_shape, net_structure, out_shape):
        super(Model, self).__init__()
        self.inner_shape = inner_shape
        self.out_shape = out_shape
        self.net_structure = net_structure
        layer = []
        for idx, hidden_unit in enumerate(self.net_structure):
            if not idx:
                layer.append(nn.Linear(inner_shape, hidden_unit))
                layer.append(nn.ReLU(inplace=True))
            else:
                layer.append(nn.Linear(self.net_structure[idx - 1], hidden_unit))
                layer.append(nn.ReLU(inplace=True))
        layer.append(nn.Linear(self.net_structure[-1], self.out_shape))
        self.net_seq = nn.Sequential(*layer)

    def forward(self, input_tensor):
        return self.net_seq(input_tensor)


if __name__ == '__main__':
    net = Model(3, [128, 64, 32], 1)
    x = torch.randn((10, 3))
    y = torch.randn((10, 1))
    opt = torch.optim.Adam(lr=1e-1, params=net.parameters())
    loss = torch.nn.L1Loss()

    opt.zero_grad()
    out = net(x)
    l1_loss = loss(out, y)
    l1_loss.backward()
    opt.step()
    print(net.net_seq[0].weight.grad_fn)
