"""
Конвертация RepVGG: multi-branch (train) -> single-path (deploy).
Загружает чекпоинт, сливает веса, сохраняет новый чекпоинт.
Запуск: python scripts/convert.py --checkpoint checkpoints/best.pth --output checkpoints/best_deploy.pth
"""

import argparse
from pathlib import Path

import torch

from src.models import build_model
from src.models.repvgg import RepVGG, verify_reparametrization
from src.utils.config import load_config
from src.utils.metrics import count_parameters, measure_inference_time


def main() -> None:
    parser = argparse.ArgumentParser(description="Конвертация RepVGG в deploy-режим")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    if config["data"]["dataset"] == "cifar100":
        config["model"]["num_classes"] = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загружаем модель
    model = build_model(config)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    if not isinstance(model, RepVGG):
        print("Это не RepVGG, конвертация не нужна.")
        return

    # Проверяем что репараметризация корректна
    print("Проверка корректности репараметризации...")
    ok = verify_reparametrization(model)
    if not ok:
        print("ОШИБКА: репараметризация некорректна!")
        return

    # До конвертации
    params_before = count_parameters(model)
    time_before = measure_inference_time(model)
    print(f"\nДо конвертации: {params_before:,} параметров, {time_before:.3f} мс/инференс")

    # Конвертируем
    model.reparametrize_model()

    # После
    params_after = count_parameters(model)
    time_after = measure_inference_time(model)
    print(f"После: {params_after:,} параметров, {time_after:.3f} мс/инференс")
    print(f"Ускорение: {time_before/time_after:.2f}x")

    # Сохраняем
    output_path = args.output
    if output_path is None:
        p = Path(args.checkpoint)
        output_path = str(p.parent / f"{p.stem}_deploy{p.suffix}")

    deploy_ckpt = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "deploy": True,
        "original_checkpoint": args.checkpoint,
    }
    torch.save(deploy_ckpt, output_path)
    print(f"\nDeploy-чекпоинт сохранён: {output_path}")


if __name__ == "__main__":
    main()
