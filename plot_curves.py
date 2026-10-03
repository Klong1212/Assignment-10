"""Plot loss and IoU curves from history.csv (with TensorBoard fallback).

Saves a dual-panel figure to assets/loss_curve.png highlighting the best epoch.

Usage:
    python plot_curves.py --history-csv checkpoints/unet_lane/history.csv \
        --output assets/loss_curve.png
"""
import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib.pyplot as plt
import numpy as np


def load_from_csv(csv_path):
    import csv
    epochs, train_losses, val_losses, train_ious, val_ious = [], [], [], [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))
            train_ious.append(float(row["train_iou"]))
            val_ious.append(float(row["val_iou"]))
    return {
        "epoch": np.array(epochs),
        "train_loss": np.array(train_losses),
        "val_loss": np.array(val_losses),
        "train_iou": np.array(train_ious),
        "val_iou": np.array(val_ious),
    }


def load_from_tensorboard(log_dir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(str(log_dir))
    acc.Reload()

    def get_scalars(tag):
        events = acc.Scalars(tag)
        return np.array([e.step for e in events]), np.array([e.value for e in events])

    e1, tr_l = get_scalars("Loss/train")
    _, val_l = get_scalars("Loss/val")
    _, tr_iou = get_scalars("IoU/train")
    _, val_iou = get_scalars("IoU/val")
    return {
        "epoch": e1,
        "train_loss": tr_l,
        "val_loss": val_l,
        "train_iou": tr_iou,
        "val_iou": val_iou,
    }


def plot_curves(data, output_path):
    epochs = data["epoch"]
    train_loss = data["train_loss"]
    val_loss = data["val_loss"]
    train_iou = data["train_iou"]
    val_iou = data["val_iou"]

    best_idx = int(np.argmin(val_loss))
    best_epoch = epochs[best_idx]
    best_val_loss = val_loss[best_idx]
    best_val_iou = val_iou[best_idx]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

    # 1. Loss Panel
    ax1.plot(epochs, train_loss, label="Train Loss", color="#1f77b4", linewidth=2)
    ax1.plot(epochs, val_loss, label="Val Loss", color="#ff7f0e", linewidth=2)
    ax1.scatter([best_epoch], [best_val_loss], color="#d62728", s=80, zorder=5,
                label=f"Best Val Loss ({best_val_loss:.4f} @ Ep {best_epoch})")
    ax1.axvline(best_epoch, color="#d62728", linestyle="--", alpha=0.5)
    ax1.set_title("Training & Validation Loss (BCE + Dice)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # 2. IoU Panel
    ax2.plot(epochs, train_iou, label="Train IoU", color="#2ca02c", linewidth=2)
    ax2.plot(epochs, val_iou, label="Val IoU", color="#9467bd", linewidth=2)
    ax2.scatter([best_epoch], [best_val_iou], color="#d62728", s=80, zorder=5,
                label=f"Val IoU at Best Loss ({best_val_iou:.4f})")
    ax2.axvline(best_epoch, color="#d62728", linestyle="--", alpha=0.5)
    ax2.set_title("Training & Validation IoU (48x48)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Mean IoU", fontsize=11)
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("Custom UNet Lane Segmentation — Training Curves", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_p), bbox_inches="tight")
    plt.close()
    print(f"Saved loss curve figure to: {out_p}")
    print(f"Best epoch: {best_epoch} | Best Val Loss: {best_val_loss:.4f} | Val IoU: {best_val_iou:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Plot training and validation curves.")
    parser.add_argument("--history-csv", default="checkpoints/unet_lane/history.csv")
    parser.add_argument("--log-dir", default="runs/unet_lane")
    parser.add_argument("--output", default="assets/loss_curve.png")
    args = parser.parse_args()

    csv_path = Path(args.history_csv)
    if csv_path.exists():
        print(f"Loading metrics from {csv_path}")
        data = load_from_csv(csv_path)
    else:
        print(f"History CSV not found at {csv_path}, falling back to TensorBoard events under {args.log_dir}")
        data = load_from_tensorboard(args.log_dir)

    plot_curves(data, args.output)


if __name__ == "__main__":
    main()
