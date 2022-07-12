import torch
from torchvision.models.quantization.mobilenetv2 import mobilenet_v2
import torch_tensorrt


net = mobilenet_v2(pretrained=True).cuda().eval()
torch_script_module = torch.jit.script(net)

trt_ts_module = torch_tensorrt.compile(torch_script_module,
    inputs = [torch_tensorrt.Input( # Specify input object with shape and dtype
            min_shape=[1, 3, 112, 112],
            opt_shape=[1, 3, 224, 224],
            max_shape=[1, 3, 448, 448],
            # For static size shape=[1, 3, 224, 224]
            dtype=torch.half) # Datatype of input tensor. Allowed options torch.(float|half|int32|bool)
    ],
    enabled_precisions = {torch.half}, # Run with FP16
)

torch.jit.save(torch_script_module, "pytorch.pt")
torch.jit.save(trt_ts_module, "tensorrt.ts") # save the TRT embedded Torchscript

net.eval()
net.qconfig = torch.quantization.get_default_qconfig("fbgemm")
# net.fuse_model()
net.train()
net = torch.quantization.prepare_qat(net)
net = net.cpu().eval()
net = torch.quantization.convert(net)

torch.jit.save(net, "quantization_non_fuse.pt")

net = mobilenet_v2(pretrained=True).cuda().eval()
net.eval()
net.qconfig = torch.quantization.get_default_qconfig("fbgemm")
net.fuse_model()
net.train()
net = torch.quantization.prepare_qat(net)
net = net.cpu().eval()
net = torch.quantization.convert(net)

torch.jit.save(net, "quantization_fuse.pt")