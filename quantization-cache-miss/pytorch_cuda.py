import time

import torch

net = torch.jit.load("pytorch.pt").cuda()
image = torch.ones([1, 3, 224, 224]).cuda()

for i in range(1000):
    net(image)
