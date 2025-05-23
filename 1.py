import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


# **ASFF 模块**
class FeatureFusion(nn.Module):
    def __init__(self, ch_list):
        super().__init__()
        compress_c = 8
        self.weight_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, compress_c, 1),
                nn.ReLU()) for ch in ch_list
        ])
        self.fusion_conv = nn.Conv2d(compress_c * len(ch_list), len(ch_list), 1)
        self.out_conv = nn.Sequential(
            nn.Conv2d(ch_list[0], ch_list[0], 3, padding=1),
            nn.BatchNorm2d(ch_list[0]),
            nn.ReLU()
        )

    def forward(self, inputs):
        weight_maps = [conv(x) for conv, x in zip(self.weight_convs, inputs)]
        combined = torch.cat(weight_maps, dim=1)
        weights = self.fusion_conv(combined)
        weights = F.softmax(weights, dim=1)  # (batch, num_features, height, width)

        out = sum([x * w for x, w in zip(inputs, weights.chunk(len(inputs), dim=1))])
        out = self.out_conv(out)
        return out, weights


# **读取图片并生成不同尺度的特征图**
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

def load_and_resize_image(image_path, base_size=(64, 64)):
    transform = transforms.Compose([
        transforms.Resize(base_size),  # 统一尺寸
        transforms.ToTensor()  # 转换为 (C, H, W)
    ])

    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # (1, C, H, W)

    # **模拟不同层的输入**
    up_then_down = F.interpolate(img_tensor, scale_factor=2.0, mode='bilinear', align_corners=False)  # 先上采样
    up_then_down = F.interpolate(up_then_down, scale_factor=0.5, mode='bilinear', align_corners=False)  # 再下采样

    down_then_up = F.interpolate(img_tensor, scale_factor=0.5, mode='bilinear', align_corners=False)  # 先下采样
    down_then_up = F.interpolate(down_then_up, scale_factor=2.0, mode='bilinear', align_corners=False)  # 再上采样

    # **确保所有尺度统一到 base_size**
    feature_maps = [
        F.interpolate(img_tensor, size=base_size, mode='bilinear', align_corners=False),  # 原图
        F.interpolate(up_then_down, size=base_size, mode='bilinear', align_corners=False),  # 上采样后再下采样
        F.interpolate(down_then_up, size=base_size, mode='bilinear', align_corners=False)  # 下采样后再上采样
    ]

    return feature_maps



# **可视化特征图、权重和最终输出**
# **可视化特征图、权重和最终输出**
def visualize_asff(feature_maps, weights, output):
    scale_labels = ["Higher-Level Input (Upsample-Downsample)",
                    "Same-Level Input (Original Image)",
                    "Lower-Level Input (Downsample-Upsample)"]

    # **单独绘制每个特征图**
    for i, fmap in enumerate(feature_maps):
        img = fmap[0].mean(dim=0).detach().numpy()  # 取 batch=0 并平均通道
        plt.figure(figsize=(4, 4))
        plt.imshow(img, cmap='gray')
        plt.title(f'Feature {i + 1}: {scale_labels[i]}', fontsize=12)
        plt.axis('off')

        # **去除白边**
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        plt.show()

    # **单独绘制每个权重图**
    for i in range(len(weights[0])):
        weight_reshaped = weights[0, i].squeeze(0).detach().numpy()
        if weight_reshaped.ndim == 1:
            weight_reshaped = weight_reshaped.reshape(feature_maps[i].shape[2], feature_maps[i].shape[3])  # 调整形状

        plt.figure(figsize=(4, 4))
        plt.imshow(weight_reshaped, cmap='coolwarm')
        plt.title(f'Weight {i + 1}: {scale_labels[i]}', fontsize=12)
        plt.axis('off')

        # **去除白边**
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        plt.show()

    # **单独绘制 ASFF 融合后的最终输出**
    output_img = output[0].mean(dim=0).detach().numpy()  # 取 batch=0 并平均通道
    plt.figure(figsize=(4, 4))
    plt.imshow(output_img, cmap='gray')
    plt.title("ASFF Output (Fused Feature Map)", fontsize=12)
    plt.axis('off')

    # **去除白边**
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    plt.show()



# **指定图片路径**
image_path = "asff.jpg"  # 你的图片路径
feature_maps = load_and_resize_image(image_path)

# **运行模型**
model = FeatureFusion([3, 3, 3])
outputs, weights = model(feature_maps)

# **可视化**
visualize_asff(feature_maps, weights, outputs)
