import torch
from torchvision.models.mobilenet import mobilenet_v2
import torch_tensorrt


net = mobilenet_v2(pretrained=True).cuda().eval()
torch_script_module = torch.jit.script(net)

trt_ts_module = torch_tensorrt.compile(torch_script_module,
    inputs = [torch_tensorrt.Input( # Specify input object with shape and dtype
            min_shape=[1, 3, 112, 112],
            opt_shape=[1, 3, 224, 224],
            max_shape=[1, 3, 448, 448],
            # For static size shape=[1, 3, 224, 224]
            dtype=torch.int8) # Datatype of input tensor. Allowed options torch.(float|half|int8|int32|bool)
    ],
    enabled_precisions = {torch.int8}, # Run with FP16
)
torch.jit.save(trt_ts_module, "trt_torchscript_module.ts") # save the TRT embedded Torchscript

import time
import os


def get_model_size(net):
    torch.save(net.state_dict(), "tmp.pth")
    model_size = os.path.getsize("tmp.pth") / 1e6
    os.remove("tmp.pth")
    return model_size


float32_data = torch.ones([1, 3, 224, 224]).cuda()
int8_data = torch.ones([1, 3, 224, 224], dtype=torch.half).cuda()
start = time.time()
for i in range(1000):
    result = net(float32_data) # run inference
end = time.time()
print(f"Pytorch time cost: {end - start:.2f}ms")


start = time.time()
for i in range(1000):
    result = trt_ts_module(int8_data) # run inference
end = time.time()
print(f"TensorRT time cost: {end - start:.4f}")

