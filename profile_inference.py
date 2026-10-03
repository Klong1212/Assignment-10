"""Profile inference memory footprint, parameters, and latency on CPU and GPU.

Measures:
- Parameter count and FP32 parameter size in MB
- Checkpoint file size in MB
- Process RSS (via psutil) before model load, after load, and peak during 100 inferences
- Peak CUDA memory (allocated and reserved) if GPU is available
- Mean latency (ms/img) and FPS after warmup
- Hardware details: CPU model, System RAM, GPU name, PyTorch version
Saves results to run/val_exp/profile.json.

Usage:
    python profile_inference.py --checkpoint checkpoints/unet_lane/best.pt \
        --output-json run/val_exp/profile.json
"""
import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import psutil
import torch

from model import UNet


def get_cpu_brand():
    try:
        if os.name == "nt":
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                text=True
            ).strip()
            if out:
                return out
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


def get_system_hardware_info():
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    cpu_name = get_cpu_brand()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    return {
        "cpu_model": cpu_name,
        "total_ram_gb": ram_gb,
        "gpu_model": gpu_name,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }


def measure_device_profile(checkpoint_path, device_str, n_warmup=20, n_runs=100):
    gc.collect()
    device = torch.device(device_str)
    process = psutil.Process()

    if device_str == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # 1. RSS before model load
    rss_before_mb = process.memory_info().rss / (1024 * 1024)

    # 2. Load model
    model = UNet(in_channels=3, num_classes=1)
    if Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        state_dict = state["model_state_dict"] if "model_state_dict" in state else state
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 3. RSS after model load
    rss_after_mb = process.memory_info().rss / (1024 * 1024)

    # Model parameter stats
    param_count = sum(p.numel() for p in model.parameters())
    fp32_size_mb = (param_count * 4) / (1024 * 1024)

    # Dummy batch=1 48x48 input
    dummy_input = torch.randn(1, 3, 48, 48, device=device)

    # 4. Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_input)
            if device_str == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()

    # 5. Benchmark timed inferences & track peak RSS
    peak_rss_mb = rss_after_mb
    latencies = []

    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            if device_str == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

            current_rss = process.memory_info().rss / (1024 * 1024)
            if current_rss > peak_rss_mb:
                peak_rss_mb = current_rss

    mean_latency_ms = float(sum(latencies) / len(latencies))
    fps = 1000.0 / mean_latency_ms if mean_latency_ms > 0 else 0.0

    profile = {
        "device": device_str,
        "rss_before_load_mb": round(rss_before_mb, 2),
        "rss_after_load_mb": round(rss_after_mb, 2),
        "rss_peak_inference_mb": round(peak_rss_mb, 2),
        "mean_latency_ms": round(mean_latency_ms, 3),
        "fps": round(fps, 1),
    }

    if device_str == "cuda" and torch.cuda.is_available():
        profile["cuda_peak_allocated_mb"] = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
        profile["cuda_peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / (1024 * 1024), 2)
    else:
        profile["cuda_peak_allocated_mb"] = 0.0
        profile["cuda_peak_reserved_mb"] = 0.0

    del model, dummy_input
    gc.collect()
    if device_str == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return profile, param_count, fp32_size_mb


def main():
    parser = argparse.ArgumentParser(description="Profile UNet single-image inference memory & latency.")
    parser.add_argument("--checkpoint", default="checkpoints/unet_lane/best.pt")
    parser.add_argument("--output-json", default="run/val_exp/profile.json")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    ckpt_size_mb = (checkpoint_path.stat().st_size / (1024 * 1024)) if checkpoint_path.exists() else 0.0

    print("=" * 60)
    print("PROFILING INFERENCE MEMORY & LATENCY (Batch 1, 48x48)")
    print("=" * 60)

    hardware = get_system_hardware_info()
    print(f"CPU:        {hardware['cpu_model']}")
    print(f"System RAM: {hardware['total_ram_gb']} GB")
    print(f"GPU:        {hardware['gpu_model']}")
    print(f"PyTorch:    {hardware['torch_version']}")
    print(f"Checkpoint: {checkpoint_path} ({ckpt_size_mb:.2f} MB)")
    print("-" * 60)

    results = {
        "hardware": hardware,
        "checkpoint_file_size_mb": round(ckpt_size_mb, 2),
    }

    # Run CPU profiling
    print("\n[1/2] Profiling on CPU...")
    cpu_profile, param_count, fp32_size_mb = measure_device_profile(
        checkpoint_path, "cpu", n_warmup=args.warmup, n_runs=args.runs
    )
    results["param_count"] = param_count
    results["fp32_param_size_mb"] = round(fp32_size_mb, 2)
    results["cpu"] = cpu_profile
    print(f"  CPU Latency: {cpu_profile['mean_latency_ms']:.2f} ms ({cpu_profile['fps']:.1f} FPS)")
    print(f"  CPU Process RSS (before/after/peak): "
          f"{cpu_profile['rss_before_load_mb']} / {cpu_profile['rss_after_load_mb']} / {cpu_profile['rss_peak_inference_mb']} MB")

    # Run GPU profiling if available
    if torch.cuda.is_available():
        print("\n[2/2] Profiling on GPU (CUDA)...")
        gpu_profile, _, _ = measure_device_profile(
            checkpoint_path, "cuda", n_warmup=args.warmup, n_runs=args.runs
        )
        results["gpu"] = gpu_profile
        print(f"  GPU Latency: {gpu_profile['mean_latency_ms']:.2f} ms ({gpu_profile['fps']:.1f} FPS)")
        print(f"  GPU Process RSS (before/after/peak): "
              f"{gpu_profile['rss_before_load_mb']} / {gpu_profile['rss_after_load_mb']} / {gpu_profile['rss_peak_inference_mb']} MB")
        print(f"  CUDA VRAM Peak (allocated/reserved): "
              f"{gpu_profile['cuda_peak_allocated_mb']} / {gpu_profile['cuda_peak_reserved_mb']} MB")
    else:
        results["gpu"] = None
        print("\n[2/2] CUDA not available, skipping GPU measurement.")

    out_p = Path(args.output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved profile results to: {out_p}")


if __name__ == "__main__":
    main()
