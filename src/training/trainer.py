"""
Цикл обучения: train + val на каждой эпохе, логирование в WandB,
сохранение чекпоинтов, mixed precision
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None


class Trainer:
    """Всё обучение в одном классе: оптимизатор, scheduler, AMP, логи"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        train_cfg = config["training"]

        # Loss с опциональным label smoothing
        label_smoothing = train_cfg.get("label_smoothing", 0.0)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Оптимизатор
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=train_cfg["lr"],
            momentum=train_cfg["momentum"],
            weight_decay=train_cfg["weight_decay"],
        )

        # Scheduler
        self.epochs = train_cfg["epochs"]
        warmup = train_cfg.get("warmup_epochs", 0)
        if train_cfg["scheduler"] == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=self.epochs - warmup, eta_min=1e-6
            )
        else:
            self.scheduler = MultiStepLR(self.optimizer, milestones=[100, 150], gamma=0.1)

        self.warmup_epochs = warmup
        self.warmup_lr = train_cfg["lr"]

        # Mixed precision
        self.use_amp = config["amp"]["enabled"] and device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        # Лучший результат
        self.best_val_acc = 0.0
        self.start_epoch = 0

        # Папка для чекпоинтов
        self.save_dir = Path(config["checkpoint"]["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _warmup_lr(self, epoch: int) -> None:
        """Линейный warmup lr в первые эпохи."""
        if epoch < self.warmup_epochs:
            lr = self.warmup_lr * (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

    def train_one_epoch(self, epoch: int) -> tuple[float, float]:
        """Одна эпоха обучения. Возвращает loss, accuracy"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [train]")
        for images, targets in pbar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100.0 * correct / total:.2f}%",
            )

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self) -> tuple[float, float]:
        """Прогоняем валидацию. Возвращает loss, accuracy"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, targets in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Сохраняем чекпоинт модели"""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "config": self.config,
        }

        if self.config["checkpoint"]["save_last"]:
            torch.save(state, self.save_dir / "last.pth")

        if is_best and self.config["checkpoint"]["save_best"]:
            torch.save(state, self.save_dir / "best.pth")

    def resume_from_checkpoint(self, ckpt_path: str) -> None:
        """Продолжаем обучение с чекпоинта. Загружаем только веса модели,
        optimizer и scheduler пересоздаём с нуля и прокручиваем до нужной эпохи"""
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_val_acc = ckpt.get("best_val_acc", 0.0)
        # Сбрасываем lr к начальному и пересоздаём scheduler на полные 100 эпох
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.warmup_lr
        remaining = self.epochs - self.warmup_epochs
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=remaining, eta_min=1e-6)
        # Прокручиваем scheduler до текущей эпохи
        steps = max(0, self.start_epoch - self.warmup_epochs)
        for _ in range(steps):
            self.scheduler.step()
        print(f"Resumed from epoch {self.start_epoch}, best_val_acc={self.best_val_acc:.2f}%")
        print(f"Current lr: {self.optimizer.param_groups[0]['lr']:.6f}")

    def train(self) -> dict:
        """
        Полный цикл обучения
        Возвращает словарь с итоговыми метриками
        """
        total_start = time.time()
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
            "epoch_time": [],
        }

        for epoch in range(self.start_epoch, self.epochs):
            epoch_start = time.time()

            # Warmup
            self._warmup_lr(epoch)

            # Обучение
            train_loss, train_acc = self.train_one_epoch(epoch)

            # Валидация
            val_loss, val_acc = self.validate()

            # Шаг scheduler после warmup
            if epoch >= self.warmup_epochs:
                self.scheduler.step()

            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Сохраняем историю
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["lr"].append(current_lr)
            history["epoch_time"].append(epoch_time)

            # Логируем в WandB
            if wandb is not None and wandb.run is not None:
                wandb.log(
                    {
                        "train/loss": train_loss,
                        "train/accuracy": train_acc,
                        "val/loss": val_loss,
                        "val/accuracy": val_acc,
                        "lr": current_lr,
                        "epoch_time": epoch_time,
                        "epoch": epoch + 1,
                    }
                )

            # Лучшая модель
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            self.save_checkpoint(epoch, is_best=is_best)

            print(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"train_loss: {train_loss:.4f} | train_acc: {train_acc:.2f}% | "
                f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.2f}% | "
                f"lr: {current_lr:.6f} | time: {epoch_time:.1f}s"
            )

        total_time = time.time() - total_start
        print(f"\nОбучение завершено за {total_time/60:.1f} мин")
        print(f"Лучшая val accuracy: {self.best_val_acc:.2f}%")

        # Логируем финальные метрики
        if wandb is not None and wandb.run is not None:
            wandb.run.summary["best_val_acc"] = self.best_val_acc
            wandb.run.summary["total_time_min"] = total_time / 60

        return {
            "best_val_acc": self.best_val_acc,
            "total_time": total_time,
            "history": history,
        }
