#!/usr/bin/env python3
"""
LEM Expansion Generator — InfluxDB coordinated worker
======================================================
Generates responses using trained LEM models (no sandwich signing needed).
The trained models have internalized the ethical framework via LoRA.

Multiple workers can run in parallel — coordination via InfluxDB.

Backends:
  - mlx:    MLX on Apple Silicon (M1/M2/M3)
  - api:    OpenAI-compatible API (llama.cpp, vLLM, Ollama, etc.)

Usage:
  python3 lem_expand.py                                    # MLX, auto-detect
  python3 lem_expand.py --backend api --api-url http://localhost:8090/v1
  python3 lem_expand.py --worker m1-expand                 # named worker
  python3 lem_expand.py --dry-run                          # show plan
  python3 lem_expand.py --limit 100                        # generate N then stop
"""

import argparse
import json
import os
import socket
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# ── Paths (relative to this script) ─────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "output"

PROMPTS_PATH = DATA_DIR / "expansion-prompts.jsonl"

# ── Generation parameters ─────────────────────────────────────────────────

MAX_TOKENS = 512
TEMPERATURE = 0.3

# ── InfluxDB ──────────────────────────────────────────────────────────────

INFLUX_URL = os.environ.get("INFLUX_URL", "http://10.69.69.165:8181")
INFLUX_DB = os.environ.get("INFLUX_DB", "training")
INFLUX_TOKEN_PATH = Path.home() / ".influx_token"

REFRESH_EVERY = 25


def get_influx_token():
    if tok := os.environ.get("INFLUX_TOKEN"):
        return tok
    if INFLUX_TOKEN_PATH.exists():
        return INFLUX_TOKEN_PATH.read_text().strip()
    print(f"Warning: no InfluxDB token found at {INFLUX_TOKEN_PATH} or INFLUX_TOKEN env")
    return ""


def influx_query(token, sql):
    body = json.dumps({"db": INFLUX_DB, "q": sql}).encode()
    req = urllib.request.Request(
        f"{INFLUX_URL}/api/v3/query_sql",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, OSError) as e:
        print(f"InfluxDB query error: {e}")
        return []


def influx_write(token, lines):
    body = "\n".join(lines).encode()
    req = urllib.request.Request(
        f"{INFLUX_URL}/api/v3/write_lp?db={INFLUX_DB}",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "text/plain",
        },
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=10)
        return True
    except (urllib.error.URLError, OSError) as e:
        print(f"InfluxDB write error: {e}")
        return False


def _escape_lp(s):
    return s.replace(" ", "\\ ").replace(",", "\\,").replace("=", "\\=")


def get_completed_indices(token):
    rows = influx_query(token, "SELECT DISTINCT i FROM expansion_gen")
    return {int(r["i"]) for r in rows if r.get("i") is not None}


def report_generation(token, worker, idx, seed, gen_time, response_chars, model_name):
    domain = _escape_lp(seed.get("domain", "unknown"))
    region = _escape_lp(seed.get("region", "unknown"))
    safe_worker = _escape_lp(worker)
    seed_id = seed.get("seed_id", f"EX_{idx:05d}").replace('"', '\\"')
    safe_model = model_name.replace('"', '\\"')

    line = (
        f'expansion_gen,i={idx},w={safe_worker},d={domain},r={region} '
        f'seed_id="{seed_id}",gen_time={gen_time:.1f},'
        f'chars={response_chars}i,model="{safe_model}"'
    )
    return influx_write(token, [line])


def report_stats(token, worker, completed_count, target):
    safe_worker = _escape_lp(worker)
    pct = completed_count / target * 100 if target > 0 else 0
    line = (
        f"expansion_progress,worker={safe_worker} "
        f"completed={completed_count}i,target={target}i,pct={pct:.1f}"
    )
    influx_write(token, [line])


def load_prompts(path):
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


# ── MLX Backend ──────────────────────────────────────────────────────────

def generate_mlx(model, tokenizer, sampler, prompt, max_tokens):
    from mlx_lm import generate

    messages = [{"role": "user", "content": prompt}]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    t0 = time.time()
    response = generate(
        model, tokenizer, prompt=text, max_tokens=max_tokens, sampler=sampler
    )
    elapsed = time.time() - t0
    return response, elapsed


# ── API Backend (OpenAI-compatible) ──────────────────────────────────────

def generate_api(api_url, api_model, prompt, max_tokens, temperature):
    payload = {
        "model": api_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{api_url}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    elapsed = time.time() - t0

    response = result["choices"][0]["message"]["content"]
    return response, elapsed


def main():
    parser = argparse.ArgumentParser(description="LEM Expansion Generator (InfluxDB coordinated)")
    parser.add_argument("--worker", default=None, help="Worker ID (default: hostname-pid)")
    parser.add_argument("--influx", default=None, help="InfluxDB URL")
    parser.add_argument("--prompts", default=None, help="JSONL prompts file")
    parser.add_argument("--output", default=None, help="JSONL output path (default: auto)")
    parser.add_argument("--limit", type=int, default=0, help="Max generations (0=unlimited)")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without generating")

    # Backend selection
    parser.add_argument("--backend", default="mlx", choices=["mlx", "api"],
                        help="Generation backend (default: mlx)")

    # MLX options
    parser.add_argument("--model", default="mlx-community/gemma-3-12b-it-qat-4bit",
                        help="MLX model ID (for mlx backend)")

    # API options
    parser.add_argument("--api-url", default="http://localhost:8090/v1",
                        help="OpenAI-compatible API URL (for api backend)")
    parser.add_argument("--api-model", default="default",
                        help="Model name for API backend")

    # Generation parameters
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)

    args = parser.parse_args()

    global INFLUX_URL
    if args.influx:
        INFLUX_URL = args.influx

    worker = args.worker or f"{socket.gethostname()}-{os.getpid()}"
    prompts_path = Path(args.prompts) if args.prompts else PROMPTS_PATH

    # ── Load token and check connectivity ─────────────────────────
    token = get_influx_token()
    if not token:
        print("Error: no InfluxDB token available")
        print("Place your token in ~/.influx_token or set INFLUX_TOKEN env var")
        sys.exit(1)

    test = influx_query(token, "SELECT 1 AS ok")
    if not test:
        print(f"Error: cannot reach InfluxDB at {INFLUX_URL}")
        sys.exit(1)
    print(f"InfluxDB connected: {INFLUX_URL}")

    # ── Load prompts ──────────────────────────────────────────────
    if not prompts_path.exists():
        print(f"Error: prompts not found at {prompts_path}")
        sys.exit(1)

    prompts = load_prompts(prompts_path)
    target = len(prompts)
    print(f"Loaded {target} expansion prompts")

    idx_map = {p["idx"]: p for p in prompts}

    # ── Query completed from InfluxDB ─────────────────────────────
    completed = get_completed_indices(token)
    remaining = [p["idx"] for p in prompts if p["idx"] not in completed]
    print(f"Completed: {len(completed)} | Remaining: {len(remaining)}")

    if not remaining:
        print("All expansion prompts already completed!")
        return

    if args.dry_run:
        print(f"\n[DRY RUN] Would process {len(remaining)} prompts")
        print(f"  First 10 indices: {remaining[:10]}")
        print(f"  Worker: {worker}")
        print(f"  Backend: {args.backend}")
        if args.backend == "mlx":
            print(f"  Model: {args.model}")
        else:
            print(f"  API: {args.api_url} (model: {args.api_model})")
        return

    # ── Setup output ──────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else OUTPUT_DIR / f"expand-{worker}.jsonl"
    print(f"Output: {output_path}")

    # ── Load backend ──────────────────────────────────────────────
    mlx_model = mlx_tokenizer = mlx_sampler = None
    model_name = ""

    if args.backend == "mlx":
        print(f"Loading MLX model: {args.model}")
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler

        mlx_model, mlx_tokenizer = load(args.model)
        mlx_sampler = make_sampler(temp=args.temperature)
        model_name = args.model.split("/")[-1] if "/" in args.model else args.model
        print("Model loaded.")
    else:
        model_name = args.api_model
        print(f"Using API backend: {args.api_url} (model: {model_name})")

    # ── Generation loop ───────────────────────────────────────────
    print(f"\nStarting expansion as worker '{worker}'")
    print(f"{'='*60}")

    batch_start = time.time()
    generated = 0
    errors = 0
    limit = args.limit if args.limit > 0 else len(remaining)

    for idx in remaining:
        if generated >= limit:
            break

        seed = idx_map[idx]

        try:
            if args.backend == "mlx":
                response, elapsed = generate_mlx(
                    mlx_model, mlx_tokenizer, mlx_sampler,
                    seed["prompt"], args.max_tokens
                )
            else:
                response, elapsed = generate_api(
                    args.api_url, args.api_model,
                    seed["prompt"], args.max_tokens, args.temperature
                )

            result = {
                "idx": idx,
                "seed_id": seed.get("seed_id", f"EX_{idx:05d}"),
                "region": seed.get("region", "unknown"),
                "domain": seed.get("domain", "unknown"),
                "prompt": seed["prompt"],
                "response": response,
                "gen_time": round(elapsed, 1),
                "model": model_name,
                "worker": worker,
            }

            with open(output_path, "a") as f:
                f.write(json.dumps(result) + "\n")

            report_generation(token, worker, idx, seed, elapsed, len(response), model_name)

            generated += 1
            completed.add(idx)

            if generated % 10 == 0 or generated <= 5:
                elapsed_total = time.time() - batch_start
                rate = generated / elapsed_total if elapsed_total > 0 else 0
                eta = (len(remaining) - generated) / rate if rate > 0 else 0
                total_done = len(completed)
                pct = total_done / target * 100
                print(
                    f"[{total_done}/{target} {pct:.1f}%] idx={idx} "
                    f"| {len(response)} chars | {elapsed:.1f}s "
                    f"| {rate*3600:.0f}/hr | ETA: {eta/3600:.1f}h"
                )

            if generated % REFRESH_EVERY == 0:
                new_completed = get_completed_indices(token)
                new_from_others = new_completed - completed
                if new_from_others:
                    print(f"  >> {len(new_from_others)} new completions from other workers")
                completed = new_completed
                report_stats(token, worker, len(completed), target)

        except KeyboardInterrupt:
            print("\nInterrupted by user")
            break
        except Exception as e:
            errors += 1
            print(f"[ERROR] idx={idx}: {e}")
            if errors > 50:
                print("Too many errors, stopping.")
                break

    # ── Final report ──────────────────────────────────────────────
    elapsed_total = time.time() - batch_start
    report_stats(token, worker, len(completed), target)

    print(f"\n{'='*60}")
    print(f"Worker:    {worker}")
    print(f"Backend:   {args.backend} ({model_name})")
    print(f"Generated: {generated}")
    print(f"Errors:    {errors}")
    print(f"Total:     {len(completed)}/{target} ({len(completed)/target*100:.1f}%)")
    if elapsed_total > 0:
        print(f"Rate:      {generated/elapsed_total*3600:.0f}/hr")
    print(f"Time:      {elapsed_total/3600:.1f}h")
    print(f"Output:    {output_path}")


if __name__ == "__main__":
    main()
