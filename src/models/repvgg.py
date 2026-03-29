"""
RepVGG: Making VGG-style ConvNets Great Again
Ядро проекта - реализация RepVGG блока с репараметризацией
При обучении: три параллельные ветки (3x3, 1x1, identity)
При инференсе: одна 3x3 свёртка (после слияния весов)
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class RepVGGBlock(nn.Module):
    """
    Основной блок RepVGG.
    Тренировка: out = bn3(conv3x3(x)) + bn1(conv1x1(x)) + bn_id(x)
    Инференс: out = conv3x3_merged(x) + bias
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 1,
        deploy: bool = False,
        use_identity: bool = True,
        use_1x1: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.deploy = deploy
        self.use_identity = use_identity
        self.use_1x1 = use_1x1

        if deploy:
            # Инференс-режим: одна свёртка, всё уже слито
            self.reparam_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups,
                bias=True,
            )
        else:
            # Тренировочный режим: три ветки

            # Ветка 3x3 (всегда есть)
            self.conv3x3 = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups,
                bias=False,
            )
            self.bn3x3 = nn.BatchNorm2d(out_channels)

            # Ветка 1x1
            if use_1x1:
                self.conv1x1 = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    groups=groups,
                    bias=False,
                )
                self.bn1x1 = nn.BatchNorm2d(out_channels)
            else:
                self.conv1x1 = None
                self.bn1x1 = None

            # Identity ветка - только когда каналы совпадают и stride=1
            if use_identity and in_channels == out_channels and stride == 1:
                self.bn_identity = nn.BatchNorm2d(out_channels)
            else:
                self.bn_identity = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.relu(self.reparam_conv(x))

        # Суммируем все ветки
        out = self.bn3x3(self.conv3x3(x))

        if self.conv1x1 is not None:
            out = out + self.bn1x1(self.conv1x1(x))

        if self.bn_identity is not None:
            out = out + self.bn_identity(x)

        return self.relu(out)

    def _fuse_conv_bn(
        self, conv: nn.Conv2d, bn: nn.BatchNorm2d
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Сливаем conv + bn в одну свёртку
        BN: y = gamma * (x - mean) / sqrt(var + eps) + beta
        Если перед ним conv с весами W: получаем новые W' и b'
        """
        gamma = bn.weight
        beta = bn.bias
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps

        # std = sqrt(var + eps)
        std = torch.sqrt(var + eps)

        # Масштабируем веса свёртки
        # W' = gamma / std * W  (поканально)
        scale = (gamma / std).reshape(-1, 1, 1, 1)
        fused_weight = conv.weight * scale

        # b' = beta - gamma * mean / std
        fused_bias = beta - gamma * mean / std

        return fused_weight, fused_bias

    def _get_identity_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Превращаем identity + BN в эквивалентную свёртку 3x3
        Identity - это как conv 1x1 с единичной матрицей
        """
        assert self.bn_identity is not None

        # Создаём единичную свёртку: input_dim/groups каналов на группу
        channels_per_group = self.in_channels // self.groups
        identity_kernel = torch.zeros(
            self.in_channels,
            channels_per_group,
            1,
            1,
            device=self.bn_identity.weight.device,
        )
        # Заполняем диагональ единицами
        for i in range(self.in_channels):
            identity_kernel[i, i % channels_per_group, 0, 0] = 1.0

        # Создаём фейковую conv для слияния с BN
        # не добавляем в модуль, просто используем веса
        identity_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=1,
            groups=self.groups,
            bias=False,
        ).to(identity_kernel.device)
        identity_conv.weight.data = identity_kernel

        return self._fuse_conv_bn(identity_conv, self.bn_identity)

    def _pad_1x1_to_3x3(self, kernel: torch.Tensor) -> torch.Tensor:
        """Паддим ядро 1x1 до 3x3 нулями по краям."""
        return F.pad(kernel, [1, 1, 1, 1])

    def reparametrize(self) -> None:
        """
        Сливаем веса трёх веток в одну 3x3 свёртку
        Вызывается один раз перед инференсом
        После этого блок работает как обычная conv3x3 + relu
        """
        if self.deploy:
            return

        # 1. Ветка 3x3: сливаем conv + bn
        w3, b3 = self._fuse_conv_bn(self.conv3x3, self.bn3x3)

        # 2. Ветка 1x1: сливаем conv + bn, паддим до 3x3
        if self.conv1x1 is not None:
            w1, b1 = self._fuse_conv_bn(self.conv1x1, self.bn1x1)
            w1 = self._pad_1x1_to_3x3(w1)
        else:
            w1 = torch.zeros_like(w3)
            b1 = torch.zeros_like(b3)

        # 3. Identity ветка: BN -> conv 1x1 -> паддим до 3x3
        if self.bn_identity is not None:
            w_id, b_id = self._get_identity_kernel_bias()
            w_id = self._pad_1x1_to_3x3(w_id)
        else:
            w_id = torch.zeros_like(w3)
            b_id = torch.zeros_like(b3)

        # Складываем всё
        merged_weight = w3 + w1 + w_id
        merged_bias = b3 + b1 + b_id

        # Создаём итоговую свёртку
        self.reparam_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            groups=self.groups,
            bias=True,
        )
        self.reparam_conv.weight.data = merged_weight
        self.reparam_conv.bias.data = merged_bias

        # Удаляем тренировочные ветки - они больше не нужны
        self.__delattr__("conv3x3")
        self.__delattr__("bn3x3")
        if hasattr(self, "conv1x1") and self.conv1x1 is not None:
            self.__delattr__("conv1x1")
            self.__delattr__("bn1x1")
        if hasattr(self, "bn_identity") and self.bn_identity is not None:
            self.__delattr__("bn_identity")

        self.deploy = True


class RepVGG(nn.Module):
    """
    Сеть RepVGG целиком
    Структура: стейдж 0 (один блок) + стейджи 1-4 (несколько блоков каждый) + GAP + FC
    Адаптирована под CIFAR: первый conv без stride=2, без maxpool
    """

    def __init__(
        self,
        num_blocks: list[int],
        width_multiplier: list[float],
        num_classes: int = 10,
        deploy: bool = False,
        use_identity: bool = True,
        use_1x1: bool = True,
    ):
        super().__init__()
        self.deploy = deploy
        self.use_identity = use_identity
        self.use_1x1 = use_1x1

        # Базовые ширины каналов
        self.in_channels = min(64, int(64 * width_multiplier[0]))

        # Стейдж 0: входной блок
        # Для CIFAR: stride=1, без maxpool
        self.stage0 = RepVGGBlock(
            3,
            self.in_channels,
            stride=1,
            deploy=deploy,
            use_identity=use_identity,
            use_1x1=use_1x1,
        )

        # Стейджи 1-4
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=1)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)

        # Голова классификатора
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(int(512 * width_multiplier[3]), num_classes)

    def _make_stage(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        """Собираем стейдж из нескольких RepVGG блоков."""
        blocks = []
        # Первый блок со stride
        blocks.append(
            RepVGGBlock(
                self.in_channels,
                out_channels,
                stride=stride,
                deploy=self.deploy,
                use_identity=self.use_identity,
                use_1x1=self.use_1x1,
            )
        )
        self.in_channels = out_channels
        # Остальные блоки с stride=1
        for _ in range(1, num_blocks):
            blocks.append(
                RepVGGBlock(
                    self.in_channels,
                    out_channels,
                    stride=1,
                    deploy=self.deploy,
                    use_identity=self.use_identity,
                    use_1x1=self.use_1x1,
                )
            )
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def reparametrize_model(self) -> None:
        """Переводим всю сеть в deploy-режим: сливаем все блоки"""
        for module in self.modules():
            if isinstance(module, RepVGGBlock):
                module.reparametrize()
        self.deploy = True


def build_repvgg_a0(
    num_classes: int = 10,
    deploy: bool = False,
    use_identity: bool = True,
    use_1x1: bool = True,
) -> RepVGG:
    """
    RepVGG-A0: самая маленькая конфигурация, подходит для CIFAR.
    num_blocks = [1, 2, 4, 14, 1] -> но для CIFAR уменьшим stage4
    width_multiplier = [0.75, 0.75, 0.75, 2.5]
    """
    return RepVGG(
        num_blocks=[2, 4, 14, 1],
        width_multiplier=[0.75, 0.75, 0.75, 2.5],
        num_classes=num_classes,
        deploy=deploy,
        use_identity=use_identity,
        use_1x1=use_1x1,
    )


def verify_reparametrization(model: RepVGG, input_shape: tuple = (1, 3, 32, 32)) -> bool:
    """
    Проверяем что выходы до и после репараметризации совпадают
    Прогоняем один и тот же вход через оригинал и сконвертированную модель
    На Ampere GPU TF32 снижает точность float32 свёрток, поэтому временно отключаем его для честного сравнения
    """
    # Отключаем TF32 на время проверки - иначе на Ampere GPU разница ~1e-3
    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model.eval()
    device = next(model.parameters()).device

    x = torch.randn(*input_shape).to(device)

    # Выход до репараметризации
    with torch.no_grad():
        out_train = model(x)

    # Копируем и репараметризуем
    model_deploy = copy.deepcopy(model)
    model_deploy.reparametrize_model()
    model_deploy.eval()

    with torch.no_grad():
        out_deploy = model_deploy(x)

    # Возвращаем TF32 как было
    torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
    torch.backends.cudnn.allow_tf32 = old_cudnn_tf32

    # Проверяем что выходы совпадают
    diff = (out_train - out_deploy).abs().max().item()
    match = torch.allclose(out_train, out_deploy, atol=1e-4)

    print(f"Макс. разница выходов: {diff:.2e}")
    print(f"Результат проверки: {'OK' if match else 'ОШИБКА!'}")

    return match
