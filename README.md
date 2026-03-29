# RepVGG Reproduction

Воспроизведение статьи "RepVGG: Making VGG-style ConvNets Great Again" (CVPR 2021) на CIFAR-10.

## Установка

```bash
git clone <repo-url>
cd repvgg-reproduction

# Через uv
uv sync

# Или через pip
pip install -e .
```

## Быстрый старт

### Обучение RepVGG-A0 на CIFAR-10

```bash
python scripts/train.py --config configs/default.yaml
```

### Переопределение параметров через CLI

```bash
# Другая модель
python scripts/train.py --model.name resnet20

# CIFAR-100
python scripts/train.py --data.dataset cifar100

# Меньше эпох
python scripts/train.py --training.epochs 10

# Без WandB
python scripts/train.py --wandb.enabled false

# Ablation: без identity ветки
python scripts/train.py --model.use_identity false
```

### Оценка модели

```bash
python scripts/evaluate.py --checkpoint checkpoints/best.pth
```

### Конвертация в deploy-режим

```bash
python scripts/convert.py --checkpoint checkpoints/best.pth
```

## Структура проекта

```
├── configs/default.yaml     # гиперпараметры
├── src/
│   ├── models/
│   │   ├── repvgg.py        # RepVGG + репараметризация
│   │   ├── resnet.py        # ResNet-20 для CIFAR
│   │   └── vgg.py           # VGG-16 для CIFAR
│   ├── data/cifar.py        # загрузка CIFAR + аугментации
│   ├── training/trainer.py  # цикл обучения
│   └── utils/
│       ├── config.py        # загрузка конфига
│       └── metrics.py       # параметры, FLOPs, инференс
├── scripts/
│   ├── train.py             # обучение
│   ├── evaluate.py          # оценка
│   └── convert.py           # конвертация train -> deploy
└── reports/
    ├── report.tex           # LaTeX-отчёт
    ├── report.pdf           # скомпилированный отчёт
    └── figures/             # графики экспериментов
```

## Эксперименты

1. **Верификация репараметризации** - проверка корректности слияния веток
2. **Сравнение архитектур** - RepVGG-A0 vs ResNet-20 vs VGG-16
3. **Ablation по веткам** - вклад 3x3, 1x1 и identity веток
4. **Cutout** - влияние аугментации на plain-архитектуру

## Требования

- Python >= 3.11
- PyTorch >= 2.0
- GPU с >= 8 GB VRAM (или CPU, но медленно)
