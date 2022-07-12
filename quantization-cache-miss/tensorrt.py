import time

import torch

net = torch.jit.load("./model/tensorrt.ts").cuda()
image = torch.ones([1, 3, 224, 224]).cuda()

for i in range(1000):
    net(image)
