"""
Утилиты для оценки моделей: подсчёт параметров, FLOPs, скорость инференса
"""

import time
from typing import Optional

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """Считаем обучаемые параметры модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model: nn.Module) -> int:
    """Все параметры, включая замороженные"""
    return sum(p.numel() for p in model.parameters())


def estimate_flops(model: nn.Module, input_size: tuple = (1, 3, 32, 32)) -> Optional[int]:
    """
    Оценка FLOPs через ptflops
    Если ptflops не установлен - считаем вручную (грубая оценка)
    """
    device = next(model.parameters()).device

    try:
        from ptflops import get_model_complexity_info

        macs, _ = get_model_complexity_info(
            model,
            (3, 32, 32),
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
        return int(macs * 2)  # MACs -> FLOPs (примерно)
    except ImportError:
        pass

    # Фолбэк: считаем руками для conv и linear
    total_flops = 0
    x = torch.randn(*input_size).to(device)
    hooks = []

    def conv_hook(module, inp, out):
        nonlocal total_flops
        batch = out.shape[0]
        out_h, out_w = out.shape[2], out.shape[3]
        k_h, k_w = module.kernel_size
        in_c = module.in_channels // module.groups
        out_c = module.out_channels
        # FLOPs = 2 * K*K*C_in * C_out * H_out * W_out (умножения + сложения)
        flops = 2 * k_h * k_w * in_c * out_c * out_h * out_w
        total_flops += flops

    def linear_hook(module, inp, out):
        nonlocal total_flops
        total_flops += 2 * module.in_features * module.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    model.eval()
    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

    return total_flops


def measure_inference_time(
    model: nn.Module,
    input_size: tuple = (1, 3, 32, 32),
    num_runs: int = 100,
    warmup: int = 10,
) -> float:
    """
    Замеряем среднее время инференса
    Делаем warmup прогонов, потом замеряем num_runs
    """
    device = next(model.parameters()).device
    model.eval()

    x = torch.randn(*input_size).to(device)

    # Прогрев GPU 
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Замер
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            _ = model(x)

            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # в мс

    avg_time = sum(times) / len(times)
    return avg_time


def model_summary(model: nn.Module, input_size: tuple = (1, 3, 32, 32)) -> dict:
    """Собираем все метрики модели в один словарь."""
    params = count_parameters(model)
    flops = estimate_flops(model, input_size)
    inf_time = measure_inference_time(model, input_size)

    summary = {
        "parameters": params,
        "parameters_m": params / 1e6,
        "flops": flops,
        "flops_m": flops / 1e6 if flops else None,
        "inference_time_ms": inf_time,
    }

    print(f"Параметры: {params:,} ({params/1e6:.2f}M)")
    if flops:
        print(f"FLOPs: {flops:,} ({flops/1e6:.1f}M)")
    print(f"Время инференса: {inf_time:.3f} мс")

    return summary
