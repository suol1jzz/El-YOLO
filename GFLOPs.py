from ultralytics.utils.torch_utils import get_flops
import torch
import time
from ultralytics import YOLO
from thop import profile
if __name__ == '__main__':
    model = YOLO("ASCAFPN.yaml").model
    model.info(verbose=True)  # 查看通道数变化


    # 生成归一化且设备一致的输入张量
    device = torch.device("cuda:0")  # 使用第一个 GPU（索引 0）
    model = model.to(device)
    input_tensor = torch.randn(1, 3, 640, 640).float().to(device) / 255.0  # 归一化到 [0,1]

    # 计算 FLOPs 和参数量
    flops, params = profile(model, inputs=(input_tensor,))
    gflops = flops / 1e9  # 转换为 GFLOPs

    print(f"GFLOPs: {gflops:.2f}")
    print(f"Params: {params / 1e6:.2f}M")

