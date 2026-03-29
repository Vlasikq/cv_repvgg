"""
VGG-16 адаптированный под CIFAR-10
Оригинальный VGG слишком жирный для 32x32, так что:
- убраны последние FC-слои на 4096 нейронов
- вместо них GAP + один FC
- каналов поменьше чтобы сопоставимо с RepVGG по параметрам
"""

import torch
import torch.nn as nn

# Конфиг VGG-16: число - кол-во фильтров, 'M' - maxpool
# Уменьшенная версия для CIFAR
VGG16_CIFAR_CFG: list = [
    64,
    64,
    "M",
    128,
    128,
    "M",
    256,
    256,
    256,
    "M",
    512,
    512,
    512,
    "M",
    512,
    512,
    512,
    "M",
]


class VGGCifar(nn.Module):
    """
    VGG для CIFAR. Стандартная цепочка conv-bn-relu + maxpool
    Классификатор - GAP + один линейный слой
    """

    def __init__(self, cfg: list, num_classes: int = 10, batch_norm: bool = True):
        super().__init__()
        self.features = self._make_layers(cfg, batch_norm)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)
        self._init_weights()

    def _make_layers(self, cfg: list, batch_norm: bool) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_channels = 3
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers.extend([conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)])
                else:
                    layers.extend([conv, nn.ReLU(inplace=True)])
                in_channels = v
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def build_vgg16_cifar(num_classes: int = 10) -> VGGCifar:
    """VGG-16 с batch norm, адаптированный под CIFAR"""
    return VGGCifar(VGG16_CIFAR_CFG, num_classes=num_classes, batch_norm=True)
