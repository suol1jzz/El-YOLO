import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.addmodules.Dysample import DySample
from ultralytics.nn.modules import DWSConv, ADown,Conv


class FeatureFusion(nn.Module):
    """可配置的ASFF融合模块"""

    def __init__(self, ch_list, fusion_type='asff2'):
        super().__init__()
        self.fusion_type = fusion_type
        compress_c = 8

        # 权重生成器 将所有输入的特征图通道数变为统一8
        self.weight_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, compress_c, 1),
                nn.ReLU())
            for ch in ch_list
        ])

        # 动态选择融合类型
        if fusion_type == 'asff2':
            self.fusion_conv = nn.Conv2d(compress_c * len(ch_list), len(ch_list), 1)
        elif fusion_type == 'asff3':
            self.fusion_conv = nn.Conv2d(compress_c * len(ch_list), len(ch_list), 1)
        else:
            raise NotImplementedError

        # 输出处理
        self.out_conv = nn.Sequential(
            nn.Conv2d(ch_list[0], ch_list[0], 3, padding=1),
            nn.BatchNorm2d(ch_list[0]),
            nn.ReLU())

    def forward(self, inputs):
        # 生成权重
        weight_maps = [conv(x) for conv, x in zip(self.weight_convs, inputs)]
        combined = torch.cat(weight_maps, dim=1)
        weights = self.fusion_conv(combined)
        weights = F.softmax(weights, dim=1)

        # 加权融合
        out = sum([x * w for x, w in zip(inputs, weights.chunk(len(inputs), dim=1))])
        out = self.out_conv(out)
        # print(f"Fusion:{out.shape}")
        return out


class ResUpDownSample(nn.Module):
    def __init__(self, in_ch, out_ch, mode='down'):
        super(ResUpDownSample, self).__init__()

        self.mode = mode
        self.in_ch = in_ch
        self.out_ch = out_ch

        # 下采样操作
        if mode == 'down':
            # self.main = DWSConv(in_ch, out_ch,3,2)
            self.main = ADown(in_ch, out_ch)
        # 上采样操作
        elif mode == 'up':
            self.main = nn.Sequential(
                DySample(in_ch,2, "lp"),  # 上采样
                Conv(in_ch, out_ch, 1 ,1),
            )
        else:
            raise ValueError("Mode should be 'down' or 'up'")

    def forward(self, x):
        # 主分支计算
        main = self.main(x)
        # print(f"ResUpDownSample_{self.mode}:{main.shape}")
        return main
