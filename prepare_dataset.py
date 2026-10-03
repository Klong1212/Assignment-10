"""Prepare dataset: filter lane polygons, format to YOLO-seg, split train/val, and export.

Reads annotations from ./dataset (or custom source-dir), locates images,
keeps only lane polygons (remapped to class 0), drops all other shapes and classes,
excludes images without lane polygons, and produces a fixed 80/20 train/val split.
Also produces assets/dataset_check.png as a sanity check.

Usage:
    python prepare_dataset.py --source-dir ./dataset --output-dir ./data
"""
import argparse
import os
import random
import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_images(images_dir):
    p = Path(images_dir)
    if not p.exists():
        return []
    return sorted([f for f in p.rglob("*") if f.suffix.lower() in IMG_EXTS])


def locate_source_images(source_dir, explicit_images_dir=None):
    if explicit_images_dir and Path(explicit_images_dir).exists():
        imgs = find_images(explicit_images_dir)
        if imgs:
            return explicit_images_dir

    src = Path(source_dir).resolve()
    # Check candidates inside source_dir and adjacent assignment folders
    candidates = [
        src / "images",
        src / "seg_yolo_data" / "images",
        src / "frames",
        src,
        # Known project structure fallbacks if images weren't duplicated into dataset/
        src.parent.parent / "Assignment-9 TensorBoard integration" / "frames",
        src.parent.parent / "Assignment-9 TensorBoard integration" / "bbox" / "images" / "train",
        src.parent / "frames",
        Path("C:/yolo_train_data/seg/images/train"),
    ]
    for c in candidates:
        if c.exists():
            imgs = find_images(c)
            if len(imgs) >= 100:
                return str(c)
    raise FileNotFoundError(f"Could not locate image files for dataset in {source_dir} or known fallback paths.")


def parse_lane_polygons(label_path, lane_class_id=0):
    """Extract only valid lane polygons from label file.
    
    Returns list of formatted lines with class remapped to 0:
    '0 x1 y1 x2 y2 ... xn yn'
    """
    valid_lines = []
    if not Path(label_path).exists():
        return valid_lines

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # Class + at least 3 points (6 coords)
                continue
            try:
                cid = int(parts[0])
            except ValueError:
                continue
            if cid != lane_class_id:
                continue

            # Ensure coords are floats and in valid range
            coords = [float(x) for x in parts[1:]]
            if len(coords) % 2 != 0:
                continue

            coord_str = " ".join(f"{c:.6f}" for c in coords)
            valid_lines.append(f"0 {coord_str}\n")
    return valid_lines


def generate_sanity_check_image(image_paths, label_paths, out_path, num_samples=3):
    """Draw ground truth polygons over random images and save side-by-side grid."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    paired = list(zip(image_paths, label_paths))
    random.seed(42)
    selected = random.sample(paired, min(num_samples, len(paired)))

    rendered_images = []
    for img_p, lbl_p in selected:
        img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        overlay = img.copy()

        polys = []
        with open(lbl_p, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 7 and parts[0] == "0":
                    coords = np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(-1, 2)
                    coords[:, 0] *= w
                    coords[:, 1] *= h
                    polys.append(coords.astype(np.int32))

        if polys:
            cv2.fillPoly(overlay, polys, (0, 255, 0))  # Green fill
            cv2.polylines(overlay, polys, isClosed=True, color=(0, 200, 0), thickness=2)

        blended = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
        # Add label title
        cv2.putText(blended, f"{img_p.stem} (Polygons: {len(polys)})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        # Resize thumbnail for compact side-by-side grid
        thumb = cv2.resize(blended, (640, 360))
        rendered_images.append(thumb)

    if rendered_images:
        combined = np.hstack(rendered_images)
        cv2.imwrite(str(out_path), combined)
        print(f"Sanity check saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert and split dataset to YOLO-seg layout.")
    parser.add_argument("--source-dir", default="./dataset", help="Original dataset root.")
    parser.add_argument("--images-dir", default=None, help="Directory containing source images.")
    parser.add_argument("--output-dir", default="./data", help="Output directory.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation fraction (default: 0.2 for 80/20).")
    parser.add_argument("--lane-class-id", type=int, default=0, help="Source class ID representing lane (default: 0).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    images_dir = locate_source_images(source_dir, args.images_dir)
    print(f"Reading images from: {images_dir}")

    # Locate labels
    labels_candidates = [
        source_dir / "labels" / "train",
        source_dir / "seg_yolo_data" / "labels" / "train",
        source_dir / "labels",
        source_dir / "seg_yolo_data" / "labels",
    ]
    labels_dir = None
    for cand in labels_candidates:
        if cand.exists() and any(cand.glob("*.txt")):
            labels_dir = cand
            break
    if labels_dir is None:
        raise FileNotFoundError(f"Could not locate label txt files under {source_dir}")
    print(f"Reading labels from: {labels_dir}")

    all_images = find_images(images_dir)
    print(f"Found {len(all_images)} total source images.")

    # Match each image with its label file and filter for valid lane polygons
    valid_samples = []
    excluded_no_polygon = 0

    for img_path in all_images:
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        lane_lines = parse_lane_polygons(lbl_path, lane_class_id=args.lane_class_id)
        if not lane_lines:
            excluded_no_polygon += 1
            continue
        valid_samples.append((img_path, lbl_path, lane_lines))

    print(f"Valid samples with lane polygons: {len(valid_samples)}")
    print(f"Excluded images (no lane polygon): {excluded_no_polygon}")

    if not valid_samples:
        raise RuntimeError("No valid samples with lane polygons found!")

    # Fixed 80/20 split with seed 42
    random.seed(args.seed)
    # Sort first for deterministic ordering before shuffle
    valid_samples.sort(key=lambda x: x[0].stem)
    random.shuffle(valid_samples)

    n_total = len(valid_samples)
    n_val = max(1, int(round(n_total * args.val_split)))
    n_train = n_total - n_val

    splits = {
        "train": valid_samples[:n_train],
        "val": valid_samples[n_train:]
    }
    print(f"Split results: train={len(splits['train'])} ({len(splits['train'])/n_total*100:.1f}%), "
          f"val={len(splits['val'])} ({len(splits['val'])/n_total*100:.1f}%)")

    output_dir = Path(args.output_dir)
    for sub in ["images", "labels"]:
        target_sub = output_dir / sub
        if target_sub.exists():
            shutil.rmtree(target_sub)

    for split_name in ["train", "val"]:
        (output_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    all_out_images, all_out_labels = [], []
    for split_name, samples in splits.items():
        for img_path, _, lane_lines in samples:
            out_img = output_dir / "images" / split_name / img_path.name
            out_lbl = output_dir / "labels" / split_name / f"{img_path.stem}.txt"

            shutil.copy2(img_path, out_img)
            with open(out_lbl, "w") as f:
                f.writelines(lane_lines)

            all_out_images.append(out_img)
            all_out_labels.append(out_lbl)

    print(f"Successfully copied images and labels to {output_dir}")

    # Generate sanity check visual
    generate_sanity_check_image(all_out_images, all_out_labels, Path("assets/dataset_check.png"))


if __name__ == "__main__":
    main()
