"""
Точка входа для обучения.
Запуск: python scripts/train.py --config configs/default.yaml
Можно переопределять параметры через CLI: --model.name repvgg_a0 --training.epochs 100
"""

import argparse
import random

import numpy as np
import torch

from src.data.cifar import get_dataloaders
from src.models import build_model
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.metrics import count_parameters, estimate_flops

try:
    import wandb
except ImportError:
    wandb = None


def set_seed(seed: int) -> None:
    """Фиксируем все сиды для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def override_config(config: dict, overrides: list[str]) -> dict:
    """
    Переопределяем параметры конфига из CLI.
    Формат: --training.lr 0.01 --model.name resnet20
    """
    i = 0
    while i < len(overrides):
        key = overrides[i].lstrip("-")
        if i + 1 < len(overrides):
            value = overrides[i + 1]
        else:
            break

        # Парсим вложенные ключи: training.lr -> config["training"]["lr"]
        keys = key.split(".")
        d = config
        for k in keys[:-1]:
            d = d[k]

        # Пытаемся угадать тип
        old_val = d.get(keys[-1])
        if old_val is not None:
            if isinstance(old_val, bool):
                value = value.lower() in ("true", "1", "yes")
            elif isinstance(old_val, int):
                value = int(value)
            elif isinstance(old_val, float):
                value = float(value)

        d[keys[-1]] = value
        i += 2

    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение RepVGG / baseline моделей")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Путь к last.pth для продолжения обучения")
    args, unknown = parser.parse_known_args()

    # Загружаем конфиг
    config = load_config(args.config)

    # CLI-переопределения
    if unknown:
        config = override_config(config, unknown)

    # Автоматом выставляем num_classes по датасету
    if config["data"]["dataset"] == "cifar100":
        config["model"]["num_classes"] = 100
    else:
        config["model"]["num_classes"] = 10

    # Воспроизводимость
    set_seed(config["seed"])

    # Устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Данные
    train_loader, val_loader = get_dataloaders(config)
    print(f"Датасет: {config['data']['dataset']}")
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    # Модель
    model = build_model(config)
    params = count_parameters(model)
    print(f"Модель: {config['model']['name']}")
    print(f"Параметры: {params:,} ({params/1e6:.2f}M)")

    # FLOPs
    flops = estimate_flops(model.to(device))
    if flops:
        print(f"FLOPs: {flops:,} ({flops/1e6:.1f}M)")

    # WandB
    wandb_cfg = config.get("wandb", {})
    if wandb is not None and wandb_cfg.get("enabled", False):
        run_name = f"{config['model']['name']}_{config['data']['dataset']}"
        wandb.init(
            project=wandb_cfg.get("project", "repvgg-reproduction"),
            entity=wandb_cfg.get("entity"),
            name=run_name,
            config=config,
            tags=wandb_cfg.get("tags", []),
        )

    # Обучение
    trainer = Trainer(model, train_loader, val_loader, config, device)

    # Resume если указан чекпоинт
    if args.resume:
        trainer.resume_from_checkpoint(args.resume)

    results = trainer.train()

    # Финализируем WandB
    if wandb is not None and wandb.run is not None:
        wandb.finish()

    print(f"\nИтого: best_val_acc = {results['best_val_acc']:.2f}%")


if __name__ == "__main__":
    main()
