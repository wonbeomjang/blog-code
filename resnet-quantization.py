import time
import os

import torch
import tqdm
from torch import Tensor
import torch.nn as nn
from torch.quantization import fuse_modules
from torchvision.datasets.mnist import MNIST
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
from torch.nn.quantized import FloatFunctional

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def conv2d(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
    return block


class BottleNeck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(BottleNeck, self).__init__()
        self.layer1 = conv2d(in_channels, out_channels, kernel_size)
        self.layer2 = conv2d(out_channels, out_channels, kernel_size)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        y = self.layer1(x)
        y = self.layer2(y)
        x = self.conv1x1(x)
        return x + y


class ResNet18(nn.Module):
    def __init__(self, num_classes, bottle_neck: nn.Module = BottleNeck):
        super(ResNet18, self).__init__()

        self.conv1 = conv2d(1, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, 2)

        self.layer1 = nn.Sequential(bottle_neck(64, 64), bottle_neck(64, 64), nn.MaxPool2d(2, 2))
        self.layer2 = nn.Sequential(bottle_neck(64, 128), bottle_neck(128, 128), nn.MaxPool2d(2, 2))
        self.layer3 = nn.Sequential(bottle_neck(128, 256), bottle_neck(256, 256), nn.MaxPool2d(2, 2))
        self.layer4 = nn.Sequential(bottle_neck(256, 512), bottle_neck(512, 512), nn.MaxPool2d(2, 2))

        self.avgpool = self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class QuantizableBottleNeck(BottleNeck):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super(QuantizableBottleNeck, self).__init__(in_channels, out_channels, kernel_size)
        self.float_functional = FloatFunctional()

    def fuse_model(self) -> None:
        fuse_modules(self.layer1, ["0", "1", "2"], inplace=True)
        fuse_modules(self.layer2, ["0", "1", "2"], inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        y = self.layer1(x)
        y = self.layer2(y)
        x = self.conv1x1(x)
        return self.float_functional.add(x, y)


class QuantizableResNet18(ResNet18):
    def __init__(self, num_classes: int):
        super(QuantizableResNet18, self).__init__(num_classes, QuantizableBottleNeck)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self._forward_impl(x)
        x = self.dequant(x)
        return x

    def fuse_model(self) -> None:
        fuse_modules(self.conv1, ["0", "1", "2"], inplace=True)
        for m in self.modules():
            if type(m) is QuantizableBottleNeck:
                m.fuse_model()


def train(net, data_loader):
    net = net.to(device).train()
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters())

    for images, target in tqdm.tqdm(data_loader):
        images = images.to(device)
        target = target.to(device)

        preds = net(images)
        loss = criterion(preds, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(net, data_loader):
    net = net.to(device).eval()
    num_correct = 0
    num_samples = 0
    total_time = 0
    torch.save(net.state_dict(), "tmp.pth")
    model_size = os.path.getsize("tmp.pth") / 1e6
    os.remove("tmp.pth")
    for images, target in tqdm.tqdm(data_loader):
        images = images.to(device)
        target = target.to(device)
        cur = time.time()
        preds = net(images)
        total_time += time.time() - cur

        preds = preds.argmax(1)

        num_correct += (preds == target).sum()
        num_samples += preds.size(0)

    print(f"Acc: {num_correct / num_samples:.4f}, Inference Time: {total_time / num_samples}, "
          f"Model Size: {model_size:.2f}MB")


if __name__ == "__main__":
    if not os.path.exists("./dataset"):
        os.mkdir("./dataset")

    train_loader = DataLoader(
        dataset=MNIST("./dataset", True, transforms.Compose([transforms.Resize(224), transforms.ToTensor()]), download=True),
        batch_size=64
    )
    test_loader = DataLoader(
        dataset=MNIST("./dataset", False, transforms.Compose([transforms.Resize(224), transforms.ToTensor()]), download=True),
        batch_size=1
    )
    resnet = ResNet18(10).to(device)
    for i in range(10):
        train(resnet, train_loader)
    test(resnet, test_loader)

    net = QuantizableResNet18(10)
    net.load_state_dict(resnet.state_dict())
    net.eval()
    net.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    net.fuse_model()

    net.train()
    net = torch.quantization.prepare_qat(net)
    train(net, train_loader)

    net = net.cpu().eval()
    net = torch.quantization.convert(net)

    device = torch.device("cpu")
    test(net, test_loader)


