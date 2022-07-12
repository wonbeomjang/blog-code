from torchvision.models.quantization.mobilenetv2 import mobilenet_v2
import torch
import os

net = mobilenet_v2(pretrained=True).cuda().eval()
net.eval()
net.qconfig = torch.quantization.get_default_qconfig("fbgemm")
# net.fuse_model()
net.train()
quantization = torch.quantization.prepare_qat(net)
quantization = quantization.cpu().eval()
quantization = torch.quantization.convert(quantization)

image = torch.ones([1, 3, 224, 224])

for i in range(5000):
    quantization(image)

torch.save(net.state_dict(), "tmp.pth")
model_size = os.path.getsize("tmp.pth") / 1e6
os.remove("tmp.pth")
print(model_size)