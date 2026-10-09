#!/usr/bin/env python3
"""Controlled CIFAR-10 comparison of four convolution operators.

The script trains a small VGG-style classifier while changing only the spatial
convolution used in three stages. It exports metrics, complexity estimates and
visualizations used by the accompanying presentation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.ops import deform_conv2d


VARIANTS = ("standard", "dilated", "depthwise", "deformable")
DISPLAY_NAMES = {
    "standard": "Обычная 3×3",
    "dilated": "Dilated, d=2",
    "depthwise": "Depthwise separable",
    "deformable": "Deformable (DCNv1)",
}
COLORS = {
    "standard": "#2563EB",
    "dilated": "#7C3AED",
    "depthwise": "#0F9D8A",
    "deformable": "#E76F51",
}
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


class DeformConv2dEducational(nn.Module):
    """DCNv1-style 3×3 convolution with an explicit offset predictor.

    Offsets are learned without extra supervision. The offset predictor starts
    at zero, so the layer initially samples the regular 3×3 grid. This code is
    intentionally transparent. Torchvision supplies the optimized sampling kernel.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        if kernel_size != 3 or padding != 1:
            raise ValueError("This educational implementation supports only 3×3, padding=1")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.offset = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size, padding=padding)
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        self.last_offsets: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offsets = self.offset(x)
        self.last_offsets = offsets.detach()
        if x.device.type == "mps":
            return self._grid_sample_forward(x, offsets)
        return deform_conv2d(x, offsets, self.weight, self.bias, padding=(1, 1))

    def _grid_sample_forward(self, x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """Vectorized DCNv1 sampling for backends without deform_conv2d backward."""
        batch, channels, height, width = x.shape
        kernel_points = self.kernel_size * self.kernel_size
        paired_offsets = offsets.reshape(batch, kernel_points, 2, height, width)
        paired_offsets = paired_offsets.permute(0, 3, 4, 1, 2)

        dtype = x.dtype
        ys = torch.arange(height, device=x.device, dtype=dtype)
        xs = torch.arange(width, device=x.device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        ky = torch.arange(self.kernel_size, device=x.device, dtype=dtype) - self.padding
        kx = torch.arange(self.kernel_size, device=x.device, dtype=dtype) - self.padding
        kernel_y, kernel_x = torch.meshgrid(ky, kx, indexing="ij")
        base_y = yy[None, :, :, None] + kernel_y.flatten()[None, None, None, :]
        base_x = xx[None, :, :, None] + kernel_x.flatten()[None, None, None, :]
        sample_y = base_y + paired_offsets[..., 0]
        sample_x = base_x + paired_offsets[..., 1]
        norm_y = 2 * sample_y / max(height - 1, 1) - 1
        norm_x = 2 * sample_x / max(width - 1, 1) - 1
        grid = torch.stack((norm_x, norm_y), dim=-1).reshape(batch, height, width * kernel_points, 2)
        sampled = F.grid_sample(
            x, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        ).reshape(batch, channels, height, width, kernel_points)
        weight = self.weight.reshape(self.out_channels, self.in_channels, kernel_points)
        out = torch.einsum("bchwk,ock->bohw", sampled, weight)
        return out + self.bias[None, :, None, None]


class SpatialOperator(nn.Module):
    def __init__(self, variant: str, in_channels: int, out_channels: int):
        super().__init__()
        if variant == "standard":
            self.op = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        elif variant == "dilated":
            self.op = nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2, bias=False)
        elif variant == "depthwise":
            self.op = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
            )
        elif variant == "deformable":
            self.op = DeformConv2dEducational(in_channels, out_channels)
        else:
            raise ValueError(f"Unknown variant: {variant}")
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.op(x)))


class CifarConvNet(nn.Module):
    def __init__(self, variant: str, num_classes: int = 10):
        super().__init__()
        self.variant = variant
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList(
            [
                SpatialOperator(variant, 16, 32),
                SpatialOperator(variant, 32, 64),
                SpatialOperator(variant, 64, 128),
            ]
        )
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.15)
        self.head = nn.Linear(128, num_classes)

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.stem(x)
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
            x = self.pool(x)
        return x, features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.forward_features(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(self.dropout(x))


@dataclass
class Result:
    variant: str
    accuracy: float
    macro_f1: float
    test_loss: float
    parameters: int
    macs: int
    latency_cpu_ms_per_image: float
    model_size_mb: float
    best_epoch: int
    best_val_accuracy: float
    max_epochs: int
    train_examples: int
    val_examples: int
    test_examples: int
    seed: int


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def make_loaders(
    data_dir: Path,
    batch_size: int,
    workers: int,
    train_subset: int | None,
    val_size: int,
    seed: int,
):
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    test_tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
    )
    train_aug_ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    train_eval_ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=test_tf)
    test_ds = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)

    if val_size < 1 or val_size >= len(train_aug_ds):
        raise ValueError(f"val_size must be in [1, {len(train_aug_ds) - 1}]")
    train_count = len(train_aug_ds) - val_size
    if train_subset is not None:
        train_count = min(train_subset, train_count)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(train_aug_ds), generator=generator).tolist()
    train_indices = indices[:train_count]
    val_indices = indices[train_count:train_count + val_size]
    train_ds = Subset(train_aug_ds, train_indices)
    val_ds = Subset(train_eval_ds, val_indices)

    common = dict(num_workers=workers, pin_memory=False, persistent_workers=workers > 0)
    shuffle_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=shuffle_generator, **common
    )
    val_loader = DataLoader(val_ds, batch_size=max(batch_size, 256), shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=max(batch_size, 256), shuffle=False, **common)
    return train_loader, val_loader, test_loader, test_ds


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * labels.size(0)
        correct += int((logits.argmax(1) == labels).sum().detach().cpu())
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses: list[float] = []
    truth: list[int] = []
    pred: list[int] = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        losses.append(float(F.cross_entropy(logits, labels, reduction="sum").cpu()))
        truth.extend(labels.cpu().tolist())
        pred.extend(logits.argmax(1).cpu().tolist())
    return {
        "loss": sum(losses) / len(truth),
        "accuracy": accuracy_score(truth, pred),
        "macro_f1": f1_score(truth, pred, average="macro"),
        "truth": truth,
        "pred": pred,
    }


def theoretical_macs(variant: str) -> int:
    # Multiply-accumulate operations for one 32×32 image. BatchNorm, activations,
    # pooling and deformable bilinear interpolation/indexing are excluded.
    macs = 32 * 32 * 3 * 16 * 9  # common stem
    h = 32
    for cin, cout in [(16, 32), (32, 64), (64, 128)]:
        if variant in ("standard", "dilated"):
            macs += h * h * cin * cout * 9
        elif variant == "depthwise":
            macs += h * h * (cin * 9 + cin * cout)
        elif variant == "deformable":
            macs += h * h * cin * 18 * 9  # offset predictor
            macs += h * h * cin * cout * 9  # weighted samples
        h //= 2
    macs += 128 * 10
    return int(macs)


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def benchmark_latency(model, device, batch_size: int = 32, warmup: int = 4, repeats: int = 12) -> float:
    model.eval()
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    for _ in range(warmup):
        model(x)
    sync(device)
    start = time.perf_counter()
    for _ in range(repeats):
        model(x)
    sync(device)
    elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / (repeats * batch_size)


def save_csv(path: Path, rows: Iterable[dict], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames = fieldnames or list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_training(history: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for variant in VARIANTS:
        rows = [r for r in history if r["variant"] == variant]
        if not rows:
            continue
        epochs = [r["epoch"] for r in rows]
        axes[0].plot(epochs, [100 * r["val_accuracy"] for r in rows], marker="o", ms=3,
                     label=DISPLAY_NAMES[variant], color=COLORS[variant])
        axes[1].plot(epochs, [r["val_loss"] for r in rows], marker="o", ms=3,
                     label=DISPLAY_NAMES[variant], color=COLORS[variant])
    axes[0].set(title="Точность на validation", xlabel="Эпоха", ylabel="Accuracy, %")
    axes[1].set(title="Функция потерь на validation", xlabel="Эпоха", ylabel="Cross-entropy")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)


def plot_tradeoff(results: list[Result], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    label_offsets = {
        "standard": (8, 12),
        "dilated": (8, -18),
        "depthwise": (8, 8),
        "deformable": (8, 8),
    }
    for r in results:
        ax.scatter(r.macs / 1e6, 100 * r.accuracy, s=90, color=COLORS[r.variant], zorder=3)
        ax.annotate(DISPLAY_NAMES[r.variant], (r.macs / 1e6, 100 * r.accuracy),
                    xytext=label_offsets[r.variant], textcoords="offset points", fontsize=10)
    ax.set(xlabel="Теоретические MACs на изображение, млн", ylabel="Test accuracy, %",
           title="Компромисс точности и вычислений")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)


def denormalize(image: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(CIFAR_MEAN)[:, None, None]
    std = torch.tensor(CIFAR_STD)[:, None, None]
    image = image.cpu() * std + mean
    return image.clamp(0, 1).permute(1, 2, 0).numpy()


@torch.no_grad()
def plot_feature_maps(models: dict[str, nn.Module], sample: torch.Tensor, label_name: str,
                      device: torch.device, out_path: Path) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(10.5, 12), constrained_layout=True)
    for row, variant in enumerate(VARIANTS):
        model = models[variant].to(device).eval()
        _, feats = model.forward_features(sample[None].to(device))
        axes[row, 0].imshow(denormalize(sample))
        axes[row, 0].set_title(f"Вход: {label_name}" if row == 0 else "То же изображение")
        for col, stage in enumerate((0, 2), start=1):
            heat = feats[stage][0].abs().mean(0).float().cpu().numpy()
            axes[row, col].imshow(heat, cmap="magma")
            axes[row, col].set_title(f"Средняя |активация|, stage {stage + 1}")
        axes[row, 0].set_ylabel(DISPLAY_NAMES[variant], fontsize=10, fontweight="bold")
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Карты признаков после исследуемой свёртки", fontsize=16, fontweight="bold")
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)


@torch.no_grad()
def plot_deform_offsets(model: CifarConvNet, sample: torch.Tensor, device: torch.device, out_path: Path) -> None:
    model = model.to(device).eval()
    model.forward_features(sample[None].to(device))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for i, block in enumerate(model.blocks):
        offsets = block.op.last_offsets[0].float().cpu()
        mag = offsets.reshape(9, 2, offsets.shape[-2], offsets.shape[-1]).square().sum(1).sqrt().mean(0)
        im = axes[i].imshow(mag.numpy(), cmap="viridis")
        axes[i].set_title(f"Stage {i + 1}: средний |offset|")
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)


def plot_confusions(eval_by_variant: dict[str, dict], class_names: list[str], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for ax, variant in zip(axes.flat, VARIANTS):
        ev = eval_by_variant[variant]
        ConfusionMatrixDisplay.from_predictions(
            ev["truth"], ev["pred"], display_labels=class_names, normalize="true",
            values_format=".1f", cmap="Blues", colorbar=False, ax=ax,
        )
        ax.set_title(DISPLAY_NAMES[variant])
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.suptitle("Нормированные матрицы ошибок", fontsize=16, fontweight="bold")
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)


def verify_deformable_equivalence() -> float:
    """At zero offsets, the educational layer must match ordinary conv."""
    torch.manual_seed(7)
    layer = DeformConv2dEducational(3, 5).double()
    x = torch.randn(2, 3, 8, 8, dtype=torch.double)
    with torch.no_grad():
        expected = F.conv2d(x, layer.weight, layer.bias, padding=1)
        actual = layer(x)
    return float((expected - actual).abs().max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--train-subset", type=int, default=None)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--resume", action="store_true", help="Reuse matching checkpoints if present")
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    print(f"device={device}; torch={torch.__version__}; deform-zero-error={verify_deformable_equivalence():.3e}")

    train_loader, val_loader, test_loader, test_ds = make_loaders(
        args.data_dir, args.batch_size, args.workers, args.train_subset, args.val_size, args.seed
    )
    train_examples = len(train_loader.dataset)
    val_examples = len(val_loader.dataset)
    test_examples = len(test_loader.dataset)
    history: list[dict] = []
    results: list[Result] = []
    models: dict[str, nn.Module] = {}
    eval_by_variant: dict[str, dict] = {}

    for variant in args.variants:
        set_seed(args.seed)
        # The educational DCN layer uses a vectorized grid_sample path on MPS,
        # where torchvision's native deform_conv2d backward is unavailable.
        train_device = device
        model = CifarConvNet(variant).to(train_device)
        run_id = f"{variant}_e{args.epochs}_train{train_examples}_val{val_examples}_s{args.seed}"
        best_ckpt = checkpoint_dir / f"{run_id}_best.pt"
        last_ckpt = checkpoint_dir / f"{run_id}_last.pt"
        variant_history_path = args.output_dir / f"history_{variant}.csv"
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        variant_history: list[dict] = []
        start_epoch = 1
        best_epoch = 0
        best_val_accuracy = -math.inf
        best_val_loss = math.inf

        if args.resume and last_ckpt.exists():
            state = torch.load(last_ckpt, map_location=train_device, weights_only=False)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            start_epoch = int(state["epoch"]) + 1
            best_epoch = int(state["best_epoch"])
            best_val_accuracy = float(state["best_val_accuracy"])
            best_val_loss = float(state["best_val_loss"])
            variant_history = list(state.get("history", []))
            print(f"[{variant}] resumed after epoch {start_epoch - 1} from {last_ckpt}")

        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.perf_counter()
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, train_device)
            val_ev = evaluate(model, val_loader, train_device)
            scheduler.step()
            row = {
                "variant": variant,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_ev["loss"],
                "val_accuracy": val_ev["accuracy"],
                "val_macro_f1": val_ev["macro_f1"],
                "learning_rate": scheduler.get_last_lr()[0],
                "seconds": time.perf_counter() - t0,
            }
            variant_history.append(row)
            improved = (
                val_ev["accuracy"] > best_val_accuracy
                or (
                    math.isclose(val_ev["accuracy"], best_val_accuracy)
                    and val_ev["loss"] < best_val_loss
                )
            )
            if improved:
                best_epoch = epoch
                best_val_accuracy = float(val_ev["accuracy"])
                best_val_loss = float(val_ev["loss"])
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": best_epoch,
                        "val_accuracy": best_val_accuracy,
                        "val_loss": best_val_loss,
                    },
                    best_ckpt,
                )
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_val_accuracy": best_val_accuracy,
                    "best_val_loss": best_val_loss,
                    "history": variant_history,
                },
                last_ckpt,
            )
            save_csv(variant_history_path, variant_history)
            print(
                f"[{variant}] {epoch:02d}/{args.epochs} "
                f"train={100*train_acc:.2f}% val={100*val_ev['accuracy']:.2f}% "
                f"loss={val_ev['loss']:.4f} best={100*best_val_accuracy:.2f}%@{best_epoch} "
                f"time={row['seconds']:.1f}s",
                flush=True,
            )

        if not best_ckpt.exists():
            raise RuntimeError(f"Best checkpoint was not created: {best_ckpt}")
        best_state = torch.load(best_ckpt, map_location=train_device, weights_only=False)
        model.load_state_dict(best_state["model"])
        best_epoch = int(best_state["epoch"])
        best_val_accuracy = float(best_state["val_accuracy"])
        history.extend(variant_history)

        # The test set is evaluated only once, after selecting the checkpoint on validation.
        ev = evaluate(model, test_loader, train_device)
        eval_by_variant[variant] = ev
        params = sum(p.numel() for p in model.parameters())
        size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        # CPU timing keeps the execution backend identical across variants.
        model = model.cpu()
        latency = benchmark_latency(model, torch.device("cpu"))
        results.append(
            Result(
                variant=variant,
                accuracy=float(ev["accuracy"]),
                macro_f1=float(ev["macro_f1"]),
                test_loss=float(ev["loss"]),
                parameters=params,
                macs=theoretical_macs(variant),
                latency_cpu_ms_per_image=latency,
                model_size_mb=size_mb,
                best_epoch=best_epoch,
                best_val_accuracy=best_val_accuracy,
                max_epochs=args.epochs,
                train_examples=train_examples,
                val_examples=val_examples,
                test_examples=test_examples,
                seed=args.seed,
            )
        )
        models[variant] = model
        if device.type == "mps":
            torch.mps.empty_cache()

    if set(args.variants) == set(VARIANTS):
        save_csv(args.output_dir / "metrics.csv", [asdict(r) for r in results])
        if history:
            save_csv(args.output_dir / "training_history.csv", history)
            plot_training(history, args.output_dir / "training_curves.png")
        plot_tradeoff(results, args.output_dir / "accuracy_vs_macs.png")
        plot_confusions(eval_by_variant, list(test_ds.classes), args.output_dir / "confusion_matrices.png")

        # Fixed test image selected independently of model predictions.
        sample, label = test_ds[12]
        cpu = torch.device("cpu")
        plot_feature_maps(models, sample, test_ds.classes[label], cpu, args.output_dir / "feature_maps.png")
        plot_deform_offsets(models["deformable"], sample, cpu, args.output_dir / "deform_offsets.png")

    metadata = {
        "device": str(device),
        "torch": torch.__version__,
        "platform": os.uname().sysname + " " + os.uname().machine,
        "deformable_zero_offset_max_abs_error": verify_deformable_equivalence(),
        "mac_definition": "multiply-accumulates; excludes BN, activations, pooling and DCN bilinear sampling/indexing",
        "latency_device": "CPU on Apple M4; batch=32; per-image average",
        "selection_protocol": "best checkpoint by validation accuracy; validation-loss tie-break; test evaluated once",
        "config": vars(args) | {"data_dir": str(args.data_dir), "output_dir": str(args.output_dir)},
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
