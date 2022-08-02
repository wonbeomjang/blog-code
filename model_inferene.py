import os
import torch
import torch.nn as nn

from torchvision import models


def get_num_param(model: nn.Module):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def get_model_size(model: nn.Module):
    torch.save(model.parameters(), "temp.pth")
    size = os.path.getsize("temp.pth")
    os.remove("temp.pth")
    return size


