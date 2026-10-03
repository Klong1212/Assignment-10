"""Generate before/after segmentation visual snapshots for best, median, and worst cases.

Reads run/val_exp/metrics.json to identify best, median, and worst IoU images on the val split.
For each case, generates a 3-panel comparison:
Original Image | Ground Truth Overlay | Predicted Mask Overlay
Saves:
- assets/inference_best.png
- assets/inference_median.png
- assets/inference_worst.png
- assets/inference_samples.png (combined 3-row overview)

Usage:
    python visualize.py --metrics-json run/val_exp/metrics.json
"""
import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import matplotlib.pyplot as plt
import numpy as np

from dataset import load_yolo_seg_polygons, polygons_to_mask, label_path_for_image


def make_overlay(image_rgb, mask, color_rgb=(0, 255, 0), alpha=0.45):
    """Draw semi-transparent color mask and crisp contour over image."""
    overlay = image_rgb.copy()
    binary_mask = (mask > 0).astype(bool)
    if binary_mask.any():
        colored_mask = np.zeros_like(image_rgb, dtype=np.uint8)
        colored_mask[binary_mask] = color_rgb
        overlay = np.where(binary_mask[:, :, None],
                           (image_rgb * (1 - alpha) + colored_mask * alpha).astype(np.uint8),
                           image_rgb)
        # Add thin boundary contour
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color_rgb, 2)
    return overlay


def render_case_panel(img_path, lbl_path, pred_path, iou, case_name, lane_class_id=0):
    image_bgr = cv2.imread(str(img_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]

    # Ground truth
    polygons = load_yolo_seg_polygons(lbl_path, w, h, lane_class_id=lane_class_id)
    gt_mask = polygons_to_mask(polygons, w, h)
    gt_overlay = make_overlay(image_rgb, gt_mask, color_rgb=(0, 220, 0), alpha=0.45)

    # Prediction
    pred_mask = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
    if pred_mask is None:
        pred_mask = np.zeros((h, w), dtype=np.uint8)
    elif pred_mask.shape != (h, w):
        pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    pred_overlay = make_overlay(image_rgb, pred_mask, color_rgb=(255, 140, 0), alpha=0.45)

    return image_rgb, gt_overlay, pred_overlay


def create_single_case_figure(image_rgb, gt_overlay, pred_overlay, case_name, img_name, iou, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150)
    axes[0].imshow(image_rgb)
    axes[0].set_title(f"Original Image: {img_name}", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(gt_overlay)
    axes[1].set_title("Ground Truth Lane (Green Overlay)", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(pred_overlay)
    axes[2].set_title(f"Predicted UNet Mask (Orange Overlay)\nIoU: {iou:.4f} (Detected: {'Yes' if iou >= 0.6 else 'No'})",
                      fontsize=12, fontweight="bold")
    axes[2].axis("off")

    plt.suptitle(f"{case_name.upper()} CASE — {img_name} — IoU: {iou:.4f}",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), bbox_inches="tight")
    plt.close()
    print(f"Saved {case_name} figure to: {out_path}")


def create_combined_figure(cases_data, out_path):
    fig, axes = plt.subplots(3, 3, figsize=(18, 14), dpi=150)
    for row, (case_name, img_name, iou, orig, gt, pred) in enumerate(cases_data):
        axes[row, 0].imshow(orig)
        axes[row, 0].set_title(f"[{case_name.upper()}] Original: {img_name}", fontsize=11, fontweight="bold")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(gt)
        axes[row, 1].set_title("Ground Truth (Green)", fontsize=11, fontweight="bold")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pred)
        axes[row, 2].set_title(f"UNet Prediction (Orange) | IoU = {iou:.4f}", fontsize=11, fontweight="bold")
        axes[row, 2].axis("off")

    plt.suptitle("Custom UNet Lane Segmentation — Validation Samples (Best, Median, Worst)",
                 fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), bbox_inches="tight")
    plt.close()
    print(f"Saved combined samples figure to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate qualitative before/after segmentation figures.")
    parser.add_argument("--metrics-json", default="run/val_exp/metrics.json")
    parser.add_argument("--images-dir", default="./data/images/val")
    parser.add_argument("--labels-dir", default="./data/labels/val")
    parser.add_argument("--pred-masks-dir", default="run/val_exp/masks")
    parser.add_argument("--assets-dir", default="assets")
    parser.add_argument("--lane-class-id", type=int, default=0)
    args = parser.parse_args()

    metrics_path = Path(args.metrics_json)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics JSON not found at {metrics_path}. Run evaluation.py first.")

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    per_image = metrics["per_image"]
    if not per_image:
        raise ValueError("No per-image results found in metrics JSON.")

    # Sort ascending by IoU
    per_image_sorted = sorted(per_image, key=lambda x: x["iou"])
    n = len(per_image_sorted)

    worst_item = per_image_sorted[0]
    median_item = per_image_sorted[n // 2]
    best_item = per_image_sorted[-1]

    cases = [
        ("best", best_item),
        ("median", median_item),
        ("worst", worst_item),
    ]

    assets_dir = Path(args.assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)

    combined_data = []
    for case_name, item in cases:
        img_name = item["image"]
        iou = item["iou"]
        img_p = Path(args.images_dir) / img_name
        lbl_p = Path(args.labels_dir) / f"{img_p.stem}.txt"
        pred_p = Path(args.pred_masks_dir) / f"{img_p.stem}.png"

        orig, gt, pred = render_case_panel(img_p, lbl_p, pred_p, iou, case_name, args.lane_class_id)
        out_single = assets_dir / f"inference_{case_name}.png"
        create_single_case_figure(orig, gt, pred, case_name, img_name, iou, out_single)
        combined_data.append((case_name, img_name, iou, orig, gt, pred))

    combined_out = assets_dir / "inference_samples.png"
    create_combined_figure(combined_data, combined_out)


if __name__ == "__main__":
    main()
