"""Compute the theoretical resolution ceiling for downsampling to 48x48.

For every validation image:
1. Rasterize full-resolution ground truth polygons (e.g. 1280x720).
2. Downsample to 48x48 using cv2.INTER_NEAREST (matching dataset.py).
3. Upsample back to original resolution using cv2.INTER_NEAREST (matching inference.py).
4. Compute pixel-wise IoU against the original full-resolution ground truth.

Reports mean, median, min, max IoU, and saves results to run/val_exp/ceiling.json.

Usage:
    python resolution_ceiling.py --images-dir data/images/val --labels-dir data/labels/val \
        --output-json run/val_exp/ceiling.json
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
import numpy as np

from dataset import list_images, load_yolo_seg_polygons, polygons_to_mask, label_path_for_image


def pixel_iou(pred_mask, gt_mask, eps=1e-6):
    pred = pred_mask > 0
    gt = gt_mask > 0
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union + eps)


def main():
    parser = argparse.ArgumentParser(description="Measure theoretical IoU ceiling imposed by 48x48 discretization.")
    parser.add_argument("--images-dir", default="data/images/val")
    parser.add_argument("--labels-dir", default="data/labels/val")
    parser.add_argument("--img-size", type=int, default=48)
    parser.add_argument("--lane-class-id", type=int, default=0)
    parser.add_argument("--output-json", default="run/val_exp/ceiling.json")
    args = parser.parse_args()

    images = list_images(args.images_dir)
    if not images:
        print(f"No images found under {args.images_dir}")
        return

    results = []
    for img_p in images:
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]

        lbl_p = label_path_for_image(img_p, args.labels_dir)
        polys = load_yolo_seg_polygons(lbl_p, w, h, lane_class_id=args.lane_class_id)
        gt_orig = polygons_to_mask(polys, w, h)

        # Downsample to 48x48 with INTER_NEAREST, then upsample back to (w, h) with INTER_NEAREST
        downsampled = cv2.resize(gt_orig, (args.img_size, args.img_size), interpolation=cv2.INTER_NEAREST)
        reconstructed = cv2.resize(downsampled, (w, h), interpolation=cv2.INTER_NEAREST)

        iou = pixel_iou(reconstructed, gt_orig)
        results.append({
            "image": img_p.name,
            "iou": float(iou)
        })

    ious = [r["iou"] for r in results]
    mean_iou = float(np.mean(ious))
    median_iou = float(np.median(ious))
    min_iou = float(np.min(ious))
    max_iou = float(np.max(ious))

    summary = {
        "num_images": len(ious),
        "mean_iou_ceiling": round(mean_iou, 4),
        "median_iou_ceiling": round(median_iou, 4),
        "min_iou_ceiling": round(min_iou, 4),
        "max_iou_ceiling": round(max_iou, 4),
    }

    print("=== Theoretical Resolution Ceiling (48x48 Discretization) ===")
    print(f"Images evaluated : {len(ious)}")
    print(f"Mean IoU ceiling : {mean_iou:.4f}")
    print(f"Median IoU ceiling: {median_iou:.4f}")
    print(f"Min IoU ceiling  : {min_iou:.4f}")
    print(f"Max IoU ceiling  : {max_iou:.4f}")

    out_p = Path(args.output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump({"summary": summary, "per_image": results}, f, indent=2)
    print(f"Saved results to: {out_p}")


if __name__ == "__main__":
    main()
