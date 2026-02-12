#!/usr/bin/env python3
"""
LEM Cross-Architecture Training
Train Llama 3.1 8B, Qwen 2.5 7B, and Mistral 7B with identical LEK data.
Same 160 examples, same hyperparams as Gemma 3.
"""
import subprocess, sys, time

MODELS = {
    "llama-3.1-8b": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "qwen-2.5-7b": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "mistral-7b-v0.3": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
}

DATA_DIR = "/Volumes/Data/lem/training"
BASE_ADAPTER = "/Volumes/Data/lem/adapters-cross"
BASE_FUSED = "/Volumes/Data/lem"

for name, model_path in MODELS.items():
    adapter_path = f"{BASE_ADAPTER}/{name}"
    fused_path = f"{BASE_FUSED}/LEM-{name}"

    print(f"\n{'='*60}")
    print(f"TRAINING: {name} ({model_path})")
    print(f"{'='*60}")
    t0 = time.time()

    # Train
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", model_path,
        "--train",
        "--data", DATA_DIR,
        "--fine-tune-type", "lora",
        "--mask-prompt",
        "--iters", "200",
        "--batch-size", "2",
        "--learning-rate", "1e-5",
        "--adapter-path", adapter_path,
        "--save-every", "100",
        "--steps-per-eval", "50",
        "--max-seq-length", "2048",
    ]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR training {name}")
        continue

    train_time = time.time() - t0
    print(f"\nTraining took {train_time:.0f}s")

    # Fuse
    print(f"\nFusing {name}...")
    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", model_path,
        "--adapter-path", adapter_path,
        "--save-path", fused_path,
    ]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR fusing {name}")
        continue

    total_time = time.time() - t0
    print(f"\n{name} complete in {total_time:.0f}s")
    print(f"Fused model at: {fused_path}")

print(f"\n{'='*60}")
print("ALL CROSS-ARCHITECTURE TRAINING COMPLETE")
print(f"{'='*60}")
