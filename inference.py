"""Run lane segmentation inference and save binary masks to disk.

Example:
    python inference.py --images-dir ./data/images/val \
        --checkpoint checkpoints/unet_lane/best.pt --run-name val_exp
"""
import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np
import torch

from dataset import list_images
from model import UNet


def load_model(checkpoint_path, device):
    model = UNet(in_channels=3, num_classes=1)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = state["model_state_dict"] if "model_state_dict" in state else state
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def preprocess(image_rgb, img_size):
    resized = cv2.resize(image_rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return tensor


@torch.no_grad()
def predict_mask(model, image_path, img_size, device, threshold=0.5):
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]

    tensor = preprocess(image_rgb, img_size).to(device)
    logits = model(tensor)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    mask_small = (prob > threshold).astype(np.uint8) * 255
    mask_full = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask_full


def main():
    parser = argparse.ArgumentParser(description="Lane segmentation inference -> binary masks.")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--img-size", type=int, default=48)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--run-name", default="exp")
    parser.add_argument("--output-root", default="run")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = load_model(args.checkpoint, device)

    out_dir = Path(args.output_root) / args.run_name / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(args.images_dir)
    if not images:
        print(f"No images found under {args.images_dir}")
        return

    for image_path in images:
        mask = predict_mask(model, image_path, args.img_size, device, args.threshold)
        out_path = out_dir / f"{image_path.stem}.png"
        cv2.imwrite(str(out_path), mask)

    print(f"Saved {len(images)} masks to {out_dir}")


if __name__ == "__main__":
    main()
