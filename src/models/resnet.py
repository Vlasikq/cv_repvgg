"""
ResNet-20 для CIFAR.
Классическая реализация из статьи He et al. 2015, адаптированная под 32x32
Первый conv - 3x3 без stride и maxpool
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Базовый блок ResNet: два conv 3x3 + skip connection"""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut: если размерности не совпадают, делаем проекцию
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetCIFAR(nn.Module):
    """
    ResNet для CIFAR-10/100
    Три группы слоёв с 16, 32, 64 каналами
    Первый conv: 3x3 stride=1, картинка 32x32, агрессивный даунсемплинг не нужен
    """

    def __init__(self, block: type, num_blocks: list[int], num_classes: int = 10):
        super().__init__()
        self.in_channels = 16

        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Три стейджа: 32x32 -> 16x16 -> 8x8
        self.stage1 = self._make_stage(block, 16, num_blocks[0], stride=1)
        self.stage2 = self._make_stage(block, 32, num_blocks[1], stride=2)
        self.stage3 = self._make_stage(block, 64, num_blocks[2], stride=2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        # Инициализация весов (Kaiming)
        self._init_weights()

    def _make_stage(
        self, block: type, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def build_resnet20(num_classes: int = 10) -> ResNetCIFAR:
    """ResNet-20: 3 стейджа по 3 блока = 6*3+2 = 20 слоёв"""
    return ResNetCIFAR(BasicBlock, [3, 3, 3], num_classes=num_classes)
