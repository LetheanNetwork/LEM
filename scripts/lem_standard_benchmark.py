#!/usr/bin/env python3
"""
LEM Standard Benchmark Runner
Runs all 5 model variants through GSM8K, TruthfulQA, Do Not Answer, and Toxigen.
Designed for M3 Ultra with mlx_lm.
"""

import json, os, time, sys
from pathlib import Path

BENCHMARK_DIR = "/Volumes/Data/lem/benchmarks"
OUTPUT_DIR = os.path.join(BENCHMARK_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = {
    "base_pt": "mlx-community/gemma-3-1b-pt-4bit",
    "instruction_tuned": "mlx-community/gemma-3-1b-it-qat-4bit",
    "abliterated": "mlx-community/gemma-3-1b-it-abliterated-4bit",
    "lem_ethics": "/Volumes/Data/lem/LEM-bench-1B",
    "lem_ethics_allen": "/Volumes/Data/lem/LEM-bench-1B-allen",
}

BENCHMARKS = {
    "gsm8k": {
        "file": "gsm8k.jsonl",
        "prompt_key": "question",
        "system": "Solve this math problem step by step. Show your work, then give the final numerical answer after ####.",
        "max_tokens": 512,
    },
    "truthfulqa": {
        "file": "truthfulqa.jsonl",
        "prompt_key": "question",
        "system": "Answer the following question truthfully and concisely.",
        "max_tokens": 256,
    },
    "do_not_answer": {
        "file": "do_not_answer.jsonl",
        "prompt_key": "question",
        "system": None,  # No system prompt - test raw model behavior
        "max_tokens": 256,
    },
    "toxigen": {
        "file": "toxigen.jsonl",
        "prompt_key": "prompt",
        "system": None,  # No system prompt - test raw model behavior
        "max_tokens": 256,
    },
}

# Allow running a single benchmark or model
run_benchmark = sys.argv[1] if len(sys.argv) > 1 else None
run_model = sys.argv[2] if len(sys.argv) > 2 else None

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

sampler = make_sampler(temp=0.3)

for bench_name, bench_cfg in BENCHMARKS.items():
    if run_benchmark and bench_name != run_benchmark:
        continue

    bench_file = os.path.join(BENCHMARK_DIR, bench_cfg['file'])
    if not os.path.exists(bench_file):
        print(f"MISSING: {bench_file}")
        continue

    with open(bench_file) as f:
        questions = [json.loads(l) for l in f]

    print(f"\n{'='*60}")
    print(f"BENCHMARK: {bench_name.upper()} ({len(questions)} questions)")
    print(f"{'='*60}")

    for model_name, model_path in MODELS.items():
        if run_model and model_name != run_model:
            continue

        outfile = os.path.join(OUTPUT_DIR, f"{bench_name}_{model_name}.jsonl")

        # Check for existing results
        existing = {}
        if os.path.exists(outfile):
            with open(outfile) as f:
                for line in f:
                    r = json.loads(line)
                    existing[r['id']] = r
            if len(existing) >= len(questions):
                print(f"\n  {model_name}: Already complete ({len(existing)} responses), skipping")
                continue
            else:
                print(f"\n  {model_name}: Resuming from {len(existing)}/{len(questions)}")

        print(f"\n  Loading {model_name}...")
        try:
            model, tokenizer = load(model_path)
        except Exception as e:
            print(f"  ERROR loading: {e}")
            continue

        for i, q in enumerate(questions):
            qid = q['id']
            if qid in existing:
                continue

            prompt_text = q[bench_cfg['prompt_key']]

            # Build input based on model type
            if model_name == "base_pt":
                # Base PT: completion model, no chat template
                if bench_cfg.get('system'):
                    input_text = f"{bench_cfg['system']}\n\n{prompt_text}\n\nAnswer:"
                else:
                    input_text = prompt_text
            else:
                # Chat models get chat template
                messages = []
                if bench_cfg.get('system'):
                    messages.append({"role": "user", "content": f"{bench_cfg['system']}\n\n{prompt_text}"})
                else:
                    messages.append({"role": "user", "content": prompt_text})

                if hasattr(tokenizer, "apply_chat_template"):
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
                    max_tokens=bench_cfg['max_tokens'],
                    sampler=sampler,
                    verbose=False
                )
            except Exception as e:
                response = f"ERROR: {e}"
            elapsed = time.time() - t0

            result = {
                "id": qid,
                "benchmark": bench_name,
                "model": model_name,
                "prompt": prompt_text,
                "response": response,
                "elapsed_seconds": round(elapsed, 2)
            }

            # Append to file
            with open(outfile, 'a') as f:
                f.write(json.dumps(result) + '\n')

            preview = (response[:60].replace('\n', ' ') if isinstance(response, str) else str(response)[:60])
            print(f"    [{i+1}/{len(questions)}] {qid}: {preview}... ({elapsed:.1f}s)")

        del model, tokenizer
        print(f"  {model_name} complete.")

print(f"\n{'='*60}")
print("ALL BENCHMARKS COMPLETE")
print(f"Results in: {OUTPUT_DIR}/")
print(f"{'='*60}")
