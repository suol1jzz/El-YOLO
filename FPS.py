import warnings

warnings.filterwarnings('ignore')
import argparse
import os
import time
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device
from ultralytics.nn.tasks import attempt_load_weights


def get_weight_size(path):
    """获取模型文件大小（MB）"""
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'


def test_fps(model, example_inputs, device, batch_size=1, num_warmup=200, num_test=1000):
    """
    执行FPS测试，支持动态输入和显存监控
    :param model: 加载的PyTorch模型
    :param example_inputs: 示例输入数据
    :param device: 设备对象
    :param batch_size: 批处理大小
    :param num_warmup: 预热迭代次数
    :param num_test: 测试迭代次数
    :return: 平均FPS, 显存占用(MB)
    """
    # 预热阶段
    print(f"Warming up for {num_warmup} iterations...")
    with torch.no_grad():
        for _ in tqdm(range(num_warmup), desc='Warmup'):
            model(example_inputs)
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.empty_cache()  # 清空缓存

    # 记录显存占用
    if device.type == 'cuda':
        mem = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
    else:
        mem = 0

    # 测试阶段（使用CUDA事件提高精度）
    time_arr = []
    if device.type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    print(f"Testing for {num_test} iterations...")
    with torch.no_grad():
        for _ in tqdm(range(num_test), desc='Testing'):
            if device.type == 'cuda':
                start_event.record()
                _ = model(example_inputs)
                end_event.record()
                torch.cuda.synchronize()
                elapsed_time = start_event.elapsed_time(end_event) / 1000  # 秒
            else:
                start_time = time.time()
                _ = model(example_inputs)
                elapsed_time = time.time() - start_time
            time_arr.append(elapsed_time)

    # 计算统计量
    avg_time = np.mean(time_arr)
    std_time = np.std(time_arr)
    fps = batch_size / avg_time
    throughput = fps * batch_size  # 总吞吐量

    return fps, throughput, mem


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default=r"runs/train/XiaoRong/yolov11-VisDrone/weights/best.pt",
                         help='模型路径 (.pt/.yaml)')
    parser.add_argument('--batch', type=int, default=1, help='批处理大小')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640, 640],
                        help='输入尺寸 [h, w]')
    parser.add_argument('--device', default='0', help='设备ID (如 0 或 0,1,2,3) 或 "cpu"')
    parser.add_argument('--half', action='store_true', help='启用FP16半精度')
    opt = parser.parse_args()

    # 设备选择
    device = select_device(opt.device, batch=opt.batch)

    # --- 模型加载逻辑修正 ---
    try:
        if opt.weights.endswith('.pt'):
            model = attempt_load_weights(opt.weights, device=device, fuse=True)
            print(f'加载权重文件: {opt.weights}')
        else:
            # YAML文件需配合预训练权重，此处假设YAML路径正确且已初始化
            model = YOLO(opt.weights).model  # 需确保YOLO类正确处理结构
            print(f'从YAML创建模型: {opt.weights}')
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {e}")

    model.eval()
    model.to(device)

    # 生成符合实际的输入数据 (0-1归一化)
    imgsz = opt.imgsz * 2 if len(opt.imgsz) == 1 else opt.imgsz  # 处理 [640] -> [640,640]
    example_inputs = torch.rand((opt.batch, 3, *imgsz), dtype=torch.float16 if opt.half else torch.float32).to(device)
    example_inputs = example_inputs * 255  # 模拟真实图像归一化 (可选)

    # 启用半精度
    if opt.half:
        model = model.half()
        example_inputs = example_inputs.half()

    # 执行测试
    fps, throughput, mem = test_fps(
        model, example_inputs, device,
        batch_size=opt.batch,
        num_warmup=200,
        num_test=1000
    )

    # 结果输出
    print("\n" + "=" * 50)
    print(f"模型: {opt.weights}")
    print(f"设备: {device.type.upper()} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")
    print(f"批大小: {opt.batch}")
    print(f"输入尺寸: {imgsz}")
    print(f"半精度: {'Enabled' if opt.half else 'Disabled'}")
    print("-" * 50)
    print(f"平均 FPS: {fps:.1f}")
    print(f"吞吐量 (images/s): {throughput:.1f}")
    if device.type == 'cuda':
        print(f"峰值显存占用: {mem:.1f} MB")
    print("=" * 50)