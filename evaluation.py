"""Evaluate predicted lane masks against YOLO-seg ground truth via pixel-wise IoU.

Metrics reported:
  - Detection rate: fraction of images classified "detected" (IoU > threshold, default 0.6)
  - Average IoU of detected results (average IoU over only the "detected" images)

Example:
    python evaluation.py --images-dir ./data/images/val --labels-dir ./data/labels/val \
        --pred-masks-dir run/val_exp/masks --output-json run/val_exp/metrics.json
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
    parser = argparse.ArgumentParser(description="Evaluate lane masks via pixel-wise IoU.")
    parser.add_argument("--images-dir", required=True, help="Original images (for size + filenames).")
    parser.add_argument("--labels-dir", required=True, help="YOLO-seg polygon label directory.")
    parser.add_argument("--pred-masks-dir", required=True, help="Predicted mask PNGs from inference.py.")
    parser.add_argument("--iou-threshold", type=float, default=0.6)
    parser.add_argument("--lane-class-id", type=int, default=0, help="Class ID to treat as lane ground truth.")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    images = list_images(args.images_dir)
    if not images:
        print(f"No images found under {args.images_dir}")
        return

    results = []
    for image_path in images:
        pred_path = Path(args.pred_masks_dir) / f"{image_path.stem}.png"
        if not pred_path.exists():
            print(f"Warning: missing prediction for {image_path.name}, skipping.")
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        h, w = image.shape[:2]

        label_path = label_path_for_image(image_path, args.labels_dir)
        polygons = load_yolo_seg_polygons(label_path, w, h, lane_class_id=args.lane_class_id)
        gt_mask = polygons_to_mask(polygons, w, h)

        pred_mask = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        if pred_mask is None:
            print(f"Warning: could not read prediction {pred_path}, skipping.")
            continue
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        iou = pixel_iou(pred_mask, gt_mask)
        detected = iou > args.iou_threshold
        results.append({"image": image_path.name, "iou": iou, "detected": detected})

    n = len(results)
    detected_results = [r for r in results if r["detected"]]
    detection_rate = len(detected_results) / n if n else 0.0
    avg_iou_detected = (sum(r["iou"] for r in detected_results) / len(detected_results)
                         if detected_results else 0.0)
    avg_iou_all = sum(r["iou"] for r in results) / n if n else 0.0

    summary = {
        "num_images": n,
        "num_detected": len(detected_results),
        "detection_rate": detection_rate,
        "avg_iou_detected": avg_iou_detected,
        "avg_iou_all": avg_iou_all,
        "iou_threshold": args.iou_threshold,
    }

    print("=== Lane Detection Evaluation ===")
    print(f"Images evaluated         : {n}")
    print(f"Detected (IoU > {args.iou_threshold})   : {len(detected_results)} ({detection_rate * 100:.1f}%)")
    print(f"Average IoU (detected)   : {avg_iou_detected:.4f}")
    print(f"Average IoU (all images) : {avg_iou_all:.4f}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"summary": summary, "per_image": results}, f, indent=2)
        print(f"Saved detailed results to {args.output_json}")


if __name__ == "__main__":
    main()
