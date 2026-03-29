"""
Оценка обученной модели: accuracy, скорость инференса, параметры.
Можно прогнать и train-, и deploy-версию RepVGG для сравнения.
Запуск: python scripts/evaluate.py --checkpoint checkpoints/best.pth --config configs/default.yaml
"""

import argparse
import copy

import torch

from src.data.cifar import get_dataloaders
from src.models import build_model
from src.models.repvgg import RepVGG, verify_reparametrization
from src.utils.config import load_config
from src.utils.metrics import model_summary


@torch.no_grad()
def evaluate_accuracy(model: torch.nn.Module, loader, device: torch.device) -> float:
    """Считаем accuracy на всём датасете."""
    model.eval()
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return 100.0 * correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценка модели")
    parser.add_argument("--checkpoint", type=str, required=True, help="Путь к чекпоинту")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    if config["data"]["dataset"] == "cifar100":
        config["model"]["num_classes"] = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Данные
    _, val_loader = get_dataloaders(config)

    # Модель
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print("=" * 60)
    print(f"Модель: {config['model']['name']}")
    print(f"Датасет: {config['data']['dataset']}")
    print(f"Чекпоинт: {args.checkpoint}")
    print("=" * 60)

    # Метрики train-режима
    print("\n--- Train mode (multi-branch) ---")
    acc_train_mode = evaluate_accuracy(model, val_loader, device)
    print(f"Val accuracy: {acc_train_mode:.2f}%")
    train_summary = model_summary(model)

    # Если RepVGG - делаем репараметризацию и сравниваем
    if isinstance(model, RepVGG):
        print("\n--- Проверка репараметризации ---")
        verify_reparametrization(model)

        print("\n--- Deploy mode (single-path) ---")
        model_deploy = copy.deepcopy(model)
        model_deploy.reparametrize_model()
        model_deploy = model_deploy.to(device)

        acc_deploy = evaluate_accuracy(model_deploy, val_loader, device)
        print(f"Val accuracy: {acc_deploy:.2f}%")
        deploy_summary = model_summary(model_deploy)

        # Ускорение
        speedup = train_summary["inference_time_ms"] / deploy_summary["inference_time_ms"]
        print(f"\nУскорение инференса: {speedup:.2f}x")
        print(f"Разница accuracy: {abs(acc_train_mode - acc_deploy):.4f}%")


if __name__ == "__main__":
    main()
