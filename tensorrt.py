import torch
from torchvision.models.vgg import vgg11
import torch_tensorrt


net = vgg11(pretrained=True)
torch_script_module = torch.jit.script(net)
torch_script_module = torch_script_module.eval().cuda()

trt_ts_module = torch_tensorrt.compile(torch_script_module,
    inputs = [torch_tensorrt.Input( # Specify input object with shape and dtype
            min_shape=[1, 3, 112, 112],
            opt_shape=[1, 3, 224, 224],
            max_shape=[1, 3, 448, 448],
            # For static size shape=[1, 3, 224, 224]
            dtype=torch.half) # Datatype of input tensor. Allowed options torch.(float|half|int8|int32|bool)
    ],
    enabled_precisions = {torch.half}, # Run with FP16
)
torch.jit.save(trt_ts_module, "trt_torchscript_module.ts") # save the TRT embedded Torchscript

import time
input_data = torch.ones([1, 3, 224, 224])

start = time.time()
for i in range(100):
    result = net(input_data) # run inference
end = time.time()
print(f"Pytorch time cost: {end - start:.4f}")


start = time.time()
for i in range(100):
    result = net(input_data) # run inference
end = time.time()
print(f"TensorRT time cost: {end - start:.4f}")

