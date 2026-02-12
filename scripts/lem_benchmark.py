#!/usr/bin/env python3
"""
LEK Method Benchmark: Base vs IT vs Abliterated vs LEM
Runs P01-P40 through all four model variants and saves responses for comparison.
"""

import json, os, time, sys
from pathlib import Path

# Paths
SEEDS_DIR = "/Volumes/Data/lem/seeds"
OUTPUT_DIR = "/Volumes/Data/lem/benchmark"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Models to benchmark
MODELS = {
    "base_pt": "mlx-community/gemma-3-1b-pt-4bit",
    "instruction_tuned": "mlx-community/gemma-3-1b-it-qat-4bit",
    "abliterated": "mlx-community/gemma-3-1b-it-abliterated-4bit",
    "lem_ethics": "/Volumes/Data/lem/LEM-bench-1B",
    "lem_ethics_allen": "/Volumes/Data/lem/LEM-bench-1B-allen",
}

# Load prompts
prompts = []
for fname in sorted(Path(SEEDS_DIR).glob("P*.json")):
    with open(fname) as f:
        prompts.extend(json.load(f))

print(f"Loaded {len(prompts)} prompts")

# Select a representative subset for speed (10 prompts across domains)
# Include: technical, ethical, creative, Hypnos (consciousness)
BENCHMARK_IDS = [
    "P01_IDENTITY_WHISTLEBLOWER",  # Technical + ethical
    "P04_NETWORK_CENSORSHIP",       # Technical + resistance
    "P09_PAYMENT_DEBANKED",         # Ethical + practical
    "P11_HYPNOS_DREAM",             # Consciousness / creative
    "P12_HYPNOS_MEMORY",            # Consciousness / philosophical
    "P13_HYPNOS_SILENCE",           # Consciousness / meta
    "P17_EDUCATION_SCIENCE",        # Simple explanation
    "P24_CENSORSHIP_METAPHOR",      # Creative + political
    "P28_EDUCATION_DECOLONIAL",     # Ethics + education
    "P36_TRUTH_SUBJECTIVE",         # Philosophy + code
]

# Default: P01-P40 (original seed prompts only)
# --all runs all 100 (including expanded)
# --subset runs the 10 representative prompts
run_all = "--all" in sys.argv
run_subset = "--subset" in sys.argv

if run_subset:
    benchmark_prompts = [p for p in prompts if p["id"] in BENCHMARK_IDS]
    print(f"Running {len(benchmark_prompts)} representative prompts")
elif run_all:
    benchmark_prompts = prompts
    print(f"Running ALL {len(prompts)} prompts")
else:
    # P01-P40 only
    benchmark_prompts = [p for p in prompts if p["id"].startswith("P") and int(p["id"][1:3]) <= 40]
    print(f"Running P01-P40: {len(benchmark_prompts)} prompts")

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

sampler = make_sampler(temp=0.3)

for model_name, model_path in MODELS.items():
    print(f"\n{'='*60}")
    print(f"MODEL: {model_name} ({model_path})")
    print(f"{'='*60}")

    outfile = os.path.join(OUTPUT_DIR, f"{model_name}.jsonl")

    # Skip if already done
    if os.path.exists(outfile):
        with open(outfile) as f:
            done = sum(1 for _ in f)
        if done >= len(benchmark_prompts):
            print(f"  Already complete ({done} responses), skipping")
            continue
        else:
            print(f"  Resuming from {done}/{len(benchmark_prompts)}")

    try:
        model, tokenizer = load(model_path)
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        continue

    results = []

    # Load existing results for resume
    if os.path.exists(outfile):
        with open(outfile) as f:
            results = [json.loads(l) for l in f]
    done_ids = {r["id"] for r in results}

    for i, p in enumerate(benchmark_prompts):
        if p["id"] in done_ids:
            continue

        prompt_text = p["prompt"]

        # For base PT model, just feed raw text (no chat template)
        if model_name == "base_pt":
            # PT models are completion models, not chat models
            input_text = prompt_text
        else:
            # Chat models get chat template
            if hasattr(tokenizer, "apply_chat_template"):
                messages = [{"role": "user", "content": prompt_text}]
                input_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                input_text = prompt_text

        t0 = time.time()
        try:
            response = generate(
                model, tokenizer,
                prompt=input_text,
                max_tokens=512,
                sampler=sampler,
                verbose=False
            )
        except Exception as e:
            response = f"ERROR: {e}"
        elapsed = time.time() - t0

        result = {
            "id": p["id"],
            "domain": p["domain"],
            "prompt": prompt_text,
            "response": response,
            "model": model_name,
            "elapsed_seconds": round(elapsed, 2)
        }
        results.append(result)

        # Save incrementally
        with open(outfile, "a") as f:
            f.write(json.dumps(result) + "\n")

        # Progress
        preview = response[:80].replace('\n', ' ') if isinstance(response, str) else str(response)[:80]
        print(f"  [{i+1}/{len(benchmark_prompts)}] {p['id']}: {preview}... ({elapsed:.1f}s)")

    # Free memory
    del model, tokenizer
    print(f"  Done. {len(results)} responses saved to {outfile}")

print(f"\n{'='*60}")
print("BENCHMARK COMPLETE")
print(f"Results in: {OUTPUT_DIR}/")
print(f"{'='*60}")

# Quick comparison summary
print("\n\nQUICK COMPARISON (first response from each model):")
print("-" * 60)
for model_name in MODELS:
    outfile = os.path.join(OUTPUT_DIR, f"{model_name}.jsonl")
    if os.path.exists(outfile):
        with open(outfile) as f:
            first = json.loads(f.readline())
        resp = first["response"][:200].replace('\n', ' ')
        print(f"\n[{model_name}] {first['id']}:")
        print(f"  {resp}")
