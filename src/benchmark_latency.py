#!/usr/bin/env python3
"""Repeatable CPU latency benchmark for the trained CIFAR-10 models."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import torch

from experiment import CifarConvNet, VARIANTS, benchmark_latency


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-examples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()

    rows = []
    medians = {}
    for variant in VARIANTS:
        model = CifarConvNet(variant)
        checkpoint = args.output_dir / "checkpoints" / (
            f"{variant}_e{args.epochs}_n{args.train_examples}_s{args.seed}.pt"
        )
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        values = [
            benchmark_latency(model, torch.device("cpu"), batch_size=32, warmup=10, repeats=args.repeats)
            for _ in range(args.trials)
        ]
        medians[variant] = statistics.median(values)
        for i, value in enumerate(values, 1):
            rows.append({"variant": variant, "trial": i, "latency_cpu_ms_per_image": value})
        print(variant, "median", medians[variant], "trials", values)

    with (args.output_dir / "latency_trials.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metrics_path = args.output_dir / "metrics.csv"
    with metrics_path.open(newline="", encoding="utf-8") as f:
        metrics = list(csv.DictReader(f))
        fieldnames = list(metrics[0])
    for row in metrics:
        row["latency_cpu_ms_per_image"] = f"{medians[row['variant']]:.9f}"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


if __name__ == "__main__":
    main()
