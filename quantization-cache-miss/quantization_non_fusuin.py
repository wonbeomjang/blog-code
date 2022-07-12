import time

import torch

net = torch.jit.load("./model/quantization_non_fuse.pt")
image = torch.ones([1, 3, 224, 224])

for i in range(1000):
    net(image)
