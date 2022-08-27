import os
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
from ptflops import get_model_complexity_info


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def conv2d(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0,
           num_block: int = 1):
    block = []
    for i in range(num_block):
        block += [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding), nn.ReLU(),
                  nn.BatchNorm2d(out_channels)]

    return block


class Model1(nn.Module):
    def __init__(self, scale: int = 1):
        super(Model1, self).__init__()
        self.feature = nn.Sequential(
            *conv2d(3, 3, 3, 1, 1, 21),
            nn.MaxPool2d(2),
            *conv2d(3, 3, 3, 1, 1, 22),
            nn.MaxPool2d(2),
            *conv2d(3, 3, 3, 1, 1, 22),
            nn.MaxPool2d(2),
            *conv2d(3, 3, 3, 1, 1, 22),
            nn.MaxPool2d(2),
            *conv2d(3, 3, 3, 1, 1, 22),
            nn.MaxPool2d(2),
            *conv2d(3, 3, 3, 1, 1, 22),
        )

    def forward(self, x):
        x = self.feature(x)
        return x


class Model2(nn.Module):
    def __init__(self, scale: int = 1):
        super(Model2, self).__init__()
        self.feature = nn.Sequential(
            *conv2d(3, 3, 15, 1, 7),
            nn.MaxPool2d(2),
            *conv2d(3, 3, 15, 1, 7),
            nn.MaxPool2d(2),
            *conv2d(3, 3, 15, 1, 7),
            nn.MaxPool2d(2),
            *conv2d(3, 3, 15, 1, 7),
            nn.MaxPool2d(2),
            *conv2d(3, 3, 15, 1, 7),
            nn.MaxPool2d(2),
            *conv2d(3, 3, 15, 1, 7),
        )

    def forward(self, x):
        x = self.feature(x)
        return x


if __name__ == "__main__":
    image = torch.rand((1, 3, 224, 224))
    net1 = Model1()
    net2 = Model2()

    macs1, params1 = get_model_complexity_info(net1, (3, 224, 224))
    macs2, params2 = get_model_complexity_info(net2, (3, 224, 224))
    print(macs1, params1)
    print(macs2, params2)

    cost = 0
    for i in range(100):
        cur = time.time_ns()
        net2(image)
        cost += time.time_ns() - cur

    print(cost / 100)
    cost = 0
    for i in range(100):
        cur = time.time_ns()
        net2(image)
        cost += time.time_ns() - cur
    print(cost / 100)
