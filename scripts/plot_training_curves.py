#!/usr/bin/env python3
"""Plot training curves from CSV files produced by train_hierarchical.py.

Usage:
    python scripts/plot_training_curves.py --model-dir models/hierarchical
    python scripts/plot_training_curves.py --model-dir models/hierarchical --save
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_curve(csv_path: str) -> dict:
    """Load a training curve CSV."""
    data = {"epoch": [], "train_loss": [], "val_loss": [], "val_ic": []}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["epoch"].append(int(row["epoch"]))
            data["train_loss"].append(float(row["train_loss"]))
            data["val_loss"].append(float(row["val_loss"]))
            ic = row.get("val_ic", "")
            data["val_ic"].append(float(ic) if ic else 0.0)
    return data


def print_summary(name: str, data: dict):
    """Print training summary to terminal."""
    n = len(data["epoch"])
    if n == 0:
        return

    best_val_idx = int(np.argmin(data["val_loss"]))
    best_ic_idx = int(np.argmax(data["val_ic"]))

    print(f"\n  {name}:")
    print(f"    Epochs trained: {n}")
    print(f"    Best val loss:  {data['val_loss'][best_val_idx]:.6f} (epoch {data['epoch'][best_val_idx]})")
    print(f"    Final val loss: {data['val_loss'][-1]:.6f}")
    print(f"    Best val IC:    {data['val_ic'][best_ic_idx]:.4f} (epoch {data['epoch'][best_ic_idx]})")
    print(f"    Final val IC:   {data['val_ic'][-1]:.4f}")

    # Check for overfitting
    if n > 5:
        last_5_train = np.mean(data["train_loss"][-5:])
        last_5_val = np.mean(data["val_loss"][-5:])
        gap = last_5_val - last_5_train
        ratio = last_5_val / max(last_5_train, 1e-10)
        if ratio > 1.5:
            print(f"    ⚠️  Overfitting detected: val/train ratio = {ratio:.2f}")
        elif ratio > 1.2:
            print(f"    ⚡ Mild overfitting: val/train ratio = {ratio:.2f}")
        else:
            print(f"    ✓ Good fit: val/train ratio = {ratio:.2f}")


def plot_curves(curves: dict, output_path: str = None):
    """Plot all training curves in a multi-panel figure."""
    if not HAS_MPL:
        print("matplotlib not available, skipping plot")
        return

    n = len(curves)
    if n == 0:
        print("No curves to plot")
        return

    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n), squeeze=False)
    fig.suptitle("Training Curves", fontsize=14, fontweight="bold")

    for idx, (name, data) in enumerate(curves.items()):
        epochs = data["epoch"]

        # Loss plot
        ax = axes[idx, 0]
        ax.plot(epochs, data["train_loss"], label="Train", color="blue", alpha=0.8)
        ax.plot(epochs, data["val_loss"], label="Val", color="red", alpha=0.8)
        best_val_idx = int(np.argmin(data["val_loss"]))
        ax.axvline(epochs[best_val_idx], color="green", linestyle="--", alpha=0.5,
                    label=f"Best (epoch {epochs[best_val_idx]})")
        ax.set_title(f"{name} — Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # IC plot
        ax = axes[idx, 1]
        ax.plot(epochs, data["val_ic"], label="Val IC", color="purple", alpha=0.8)
        ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
        best_ic_idx = int(np.argmax(data["val_ic"]))
        ax.axvline(epochs[best_ic_idx], color="green", linestyle="--", alpha=0.5,
                    label=f"Best (epoch {epochs[best_ic_idx]})")
        ax.set_title(f"{name} — Information Coefficient")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("IC")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\n  Plot saved → {output_path}")
    else:
        plt.savefig("/tmp/training_curves.png", dpi=150, bbox_inches="tight")
        print(f"\n  Plot saved → /tmp/training_curves.png")


def main():
    parser = argparse.ArgumentParser(description="Plot training curves")
    parser.add_argument("--model-dir", type=str, default="models/hierarchical",
                        help="Directory containing *_history.csv files")
    parser.add_argument("--save", action="store_true",
                        help="Save plot to model-dir/training_curves.png")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Error: {model_dir} does not exist")
        sys.exit(1)

    # Find all *_history.csv files
    csv_files = sorted(model_dir.glob("*_history.csv"))
    if not csv_files:
        print(f"No *_history.csv files found in {model_dir}")
        print("  Train with: python train_hierarchical.py")
        sys.exit(1)

    print(f"Found {len(csv_files)} training curves in {model_dir}:")
    curves = {}
    for csv_path in csv_files:
        name = csv_path.stem.replace("_history", "")
        data = load_curve(str(csv_path))
        curves[name] = data
        print_summary(name, data)

    if args.save:
        out = str(model_dir / "training_curves.png")
    else:
        out = None

    plot_curves(curves, out)


if __name__ == "__main__":
    main()
