"""Train the custom UNet for single-class lane segmentation.

Dataset layout expected (Ultralytics YOLO-seg convention):

    <data-root>/images/train/*.jpg
    <data-root>/images/val/*.jpg      (optional; auto-split from train if absent)
    <data-root>/labels/train/*.txt
    <data-root>/labels/val/*.txt

Example:
    python train.py --data-root ./data --epochs 30 --batch-size 4
"""
import argparse
import os
import random
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import LaneSegDataset
from model import UNet


def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.reshape(probs.size(0), -1)
    targets = targets.reshape(targets.size(0), -1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    """BCE handles pixel-wise classification; Dice compensates for the heavy
    background/lane class imbalance typical of thin lane markings."""

    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        return (self.bce_weight * self.bce(logits, targets)
                + (1 - self.bce_weight) * dice_loss(logits, targets))


@torch.no_grad()
def compute_iou(logits, targets, threshold=0.5, eps=1e-6):
    preds = (torch.sigmoid(logits) > threshold).float()
    preds = preds.reshape(preds.size(0), -1)
    targets = targets.reshape(targets.size(0), -1)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def build_datasets(args):
    train_images = Path(args.data_root) / "images" / "train"
    train_labels = Path(args.data_root) / "labels" / "train"
    val_images = Path(args.data_root) / "images" / "val"
    val_labels = Path(args.data_root) / "labels" / "val"

    if val_images.exists() and any(val_images.iterdir()):
        train_ds = LaneSegDataset(train_images, train_labels, args.img_size, augment=True,
                                  lane_class_id=args.lane_class_id)
        val_ds = LaneSegDataset(val_images, val_labels, args.img_size, augment=False,
                                lane_class_id=args.lane_class_id)
        return train_ds, val_ds

    # No explicit val split provided: carve one out of the train set.
    augmented = LaneSegDataset(train_images, train_labels, args.img_size, augment=True,
                               lane_class_id=args.lane_class_id)
    plain = LaneSegDataset(train_images, train_labels, args.img_size, augment=False,
                           lane_class_id=args.lane_class_id)
    n = len(augmented)
    if n == 0:
        raise RuntimeError(f"No training images found under {train_images}")

    indices = list(range(n))
    random.Random(args.seed).shuffle(indices)
    val_len = max(1, int(n * args.val_split))
    val_idx, train_idx = indices[:val_len], indices[val_len:]

    train_ds = Subset(augmented, train_idx)
    val_ds = Subset(plain, val_idx)
    return train_ds, val_ds


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_iou, n_batches = 0.0, 0.0, 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_iou += compute_iou(logits, masks)
            n_batches += 1

    n_batches = max(n_batches, 1)
    return total_loss / n_batches, total_iou / n_batches


def get_safe_log_dir(path):
    p = Path(path).resolve()
    p.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        import ctypes
        buf = ctypes.create_unicode_buffer(500)
        res = ctypes.windll.kernel32.GetShortPathNameW(str(p), buf, 500)
        if res:
            return buf.value
    return str(p)


def main():
    parser = argparse.ArgumentParser(description="Train custom UNet for lane segmentation.")
    parser.add_argument("--data-root", required=True, help="Root dir containing images/ and labels/.")
    parser.add_argument("--img-size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4, help="Recommended 2-4 for local machines.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.1, help="Used only if no images/val split exists.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lane-class-id", type=int, default=0, help="Class ID to treat as lane ground truth.")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--log-dir", default="runs")
    parser.add_argument("--run-name", default="unet_lane")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds, val_ds = build_datasets(args)
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, drop_last=False)

    model = UNet(in_channels=3, num_classes=1).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    checkpoint_dir = Path(args.checkpoint_dir) / args.run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=get_safe_log_dir(Path(args.log_dir) / args.run_name))

    history_csv_path = checkpoint_dir / "history.csv"
    with open(history_csv_path, "w") as f:
        f.write("epoch,train_loss,val_loss,train_iou,val_iou\n")

    best_val_loss = float("inf")
    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs"):
        train_loss, train_iou = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_iou = run_epoch(model, val_loader, criterion, device, optimizer=None)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("IoU/train", train_iou, epoch)
        writer.add_scalar("IoU/val", val_iou, epoch)

        with open(history_csv_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{train_iou:.6f},{val_iou:.6f}\n")

        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_iou={train_iou:.4f} | "
              f"val_loss={val_loss:.4f} val_iou={val_iou:.4f}")

        torch.save({"epoch": epoch, "model_state_dict": model.state_dict()},
                   checkpoint_dir / "last.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()},
                       checkpoint_dir / "best.pt")

    writer.close()
    print(f"Training complete. Checkpoints saved under {checkpoint_dir}")


if __name__ == "__main__":
    main()
