import torch
import torch.nn as nn
from ultralytics.nn.modules import *


class ACBiFPN(nn.Module):
    def __init__(self, c2, num_features=3):
        super(ACBiFPN, self).__init__()
        self.num_features = num_features  # 支持2或3个特征图输入
        self.epsilon = 1e-4

        # 动态初始化权重参数（权重数量=输入特征数）
        self.w = nn.Parameter(torch.ones(num_features, dtype=torch.float32), requires_grad=True)

        # 自适应通道压缩（解决Concat后通道膨胀问题）
        self.conv = Conv(c2 * 2 if num_features == 2 else c2 * 3, c2, 1, 1, 0)

    def forward(self, x):
        # 输入校验：x为2或3个特征图的列表
        assert len(x) in [2, 3], "Input must be 2 or 3 features!"
        w = self.w
        # 权重归一化（Softmax风格）
            # temperature = nn.Parameter(torch.tensor(1.0))  # 可学习温度系数
            # weights = torch.softmax(self.w / temperature, dim=0)
        weights = w / (torch.sum(w, dim=0) + self.epsilon)

        # 加权特征融合（动态处理2/3个输入）
        weighted_features = [w * feat for w, feat in zip(weights, x)]

        # 沿通道维度拼接
        fused = torch.cat(weighted_features, dim=1)
        # 通道压缩 + 激活（平衡计算量）
        fused = self.conv(fused)

        return fused
