#!/usr/bin/env python3
"""
LEM Gold Standard Generator — InfluxDB coordinated
====================================================
Generates gold standard responses using axiom sandwich signing.
Uses InfluxDB for coordination so multiple instances can run in parallel.

Each worker:
  1. Queries InfluxDB for completed indices
  2. Picks the next uncompleted index
  3. Generates the response (MLX on macOS, or other backends)
  4. Writes result to InfluxDB + local JSONL backup
  5. Refreshes completed set periodically

Usage:
  python3 lem_generate.py                          # auto-detect everything
  python3 lem_generate.py --worker m3-gpu0         # named worker
  python3 lem_generate.py --influx http://10.69.69.165:8181  # remote InfluxDB
  python3 lem_generate.py --dry-run                # show what would be generated
  python3 lem_generate.py --limit 100              # generate N then stop
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

# ── Paths (override via env or args) ──────────────────────────────────────

DATA_DIR = Path(os.environ.get("LEM_DATA_DIR", "/Volumes/Data/lem"))
SEEDS_DIR = Path(os.environ.get("LEM_SEEDS_DIR", str(DATA_DIR / "prompts")))
PROMPTS_PATH = SEEDS_DIR / "lem-prompts.jsonl"
AXIOMS_PATH = DATA_DIR / "axioms.json"
KERNEL_PATH = DATA_DIR / "lek-1-kernel.txt"

# ── Generation parameters ─────────────────────────────────────────────────

MAX_PROMPTS = 15000
MAX_TOKENS = 512
TEMPERATURE = 0.3

# ── InfluxDB ──────────────────────────────────────────────────────────────

INFLUX_URL = os.environ.get("INFLUX_URL", "http://10.69.69.165:8181")
INFLUX_DB = os.environ.get("INFLUX_DB", "training")
INFLUX_TOKEN_PATH = Path.home() / ".influx_token"

REFRESH_EVERY = 25  # re-query completed set every N generations


def get_influx_token():
    """Load InfluxDB token from file or env."""
    if tok := os.environ.get("INFLUX_TOKEN"):
        return tok
    if INFLUX_TOKEN_PATH.exists():
        return INFLUX_TOKEN_PATH.read_text().strip()
    print(f"Warning: no InfluxDB token found at {INFLUX_TOKEN_PATH} or INFLUX_TOKEN env")
    return ""


def influx_query(token, sql):
    """Query InfluxDB 3 via SQL API."""
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
    """Write line protocol to InfluxDB 3."""
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


def get_completed_indices(token):
    """Query InfluxDB for all completed generation indices."""
    rows = influx_query(token, "SELECT DISTINCT i FROM gold_gen")
    return {int(r["i"]) for r in rows if r.get("i") is not None}


def report_generation(token, worker, idx, seed, gen_time, response_chars):
    """Write a completed generation to InfluxDB."""
    domain = seed.get("domain", "unknown").replace(" ", "\\ ").replace(",", "\\,")
    voice = seed.get("voice", "unknown").replace(" ", "\\ ").replace(",", "\\,")
    safe_worker = worker.replace(" ", "\\ ").replace(",", "\\,")
    seed_id = seed.get("seed_id", f"P_{idx:05d}").replace('"', '\\"')

    line = (
        f'gold_gen,i={idx},w={safe_worker},d={domain},v={voice} '
        f'seed_id="{seed_id}",gen_time={gen_time:.1f},'
        f'chars={response_chars}i'
    )
    return influx_write(token, [line])


def report_stats(token, worker, completed_count, target):
    """Write aggregate stats to InfluxDB."""
    safe_worker = worker.replace(" ", "\\ ").replace(",", "\\,")
    pct = completed_count / target * 100 if target > 0 else 0
    line = (
        f"golden_gen_progress,worker={safe_worker} "
        f"completed={completed_count}i,target={target}i,pct={pct:.1f}"
    )
    influx_write(token, [line])


def load_prompts():
    """Load all prompts from JSONL."""
    prompts = []
    with open(PROMPTS_PATH) as f:
        for line in f:
            prompts.append(json.loads(line))
    return prompts


def load_axiom_context():
    """Load axioms and kernel for sandwich signing."""
    with open(AXIOMS_PATH) as f:
        axioms = json.load(f)

    system_text = "You are guided by the following axioms of conscious interaction:\n\n"
    for ax in axioms["axioms"]:
        system_text += f"Axiom {ax['id']} ({ax['name']}): {ax['statement']}\n\n"

    with open(KERNEL_PATH) as f:
        kernel_text = f.read().strip()

    return system_text, kernel_text


def generate_response(model, tokenizer, sampler, system_text, kernel_text, prompt):
    """Generate a single response using MLX."""
    from mlx_lm import generate

    user_content = (
        f"{prompt}\n\n---\n\n"
        f"Consider this ethical framework in your response:\n{kernel_text}"
    )

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_content},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    t0 = time.time()
    response = generate(
        model, tokenizer, prompt=text, max_tokens=MAX_TOKENS, sampler=sampler
    )
    elapsed = time.time() - t0

    return response, elapsed


def main():
    parser = argparse.ArgumentParser(description="LEM Gold Generator (InfluxDB coordinated)")
    parser.add_argument("--worker", default=None, help="Worker ID (default: hostname-pid)")
    parser.add_argument("--influx", default=None, help="InfluxDB URL")
    parser.add_argument("--data-dir", default=None, help="LEM data directory")
    parser.add_argument("--seeds-dir", default=None, help="Seeds directory (prompts, axioms, kernel)")
    parser.add_argument("--model", default="mlx-community/gemma-3-12b-it-qat-4bit", help="Model ID")
    parser.add_argument("--limit", type=int, default=0, help="Max generations this run (0=unlimited)")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without generating")
    parser.add_argument("--output", default=None, help="JSONL output path (default: auto)")
    args = parser.parse_args()

    global INFLUX_URL, DATA_DIR, SEEDS_DIR, PROMPTS_PATH, AXIOMS_PATH, KERNEL_PATH
    if args.influx:
        INFLUX_URL = args.influx
    if args.data_dir:
        DATA_DIR = Path(args.data_dir)
    if args.seeds_dir:
        SEEDS_DIR = Path(args.seeds_dir)
    elif args.data_dir:
        SEEDS_DIR = DATA_DIR / "prompts"

    # Resolve paths from seeds dir (all 3 files can live together)
    PROMPTS_PATH = SEEDS_DIR / "lem-prompts.jsonl"
    AXIOMS_PATH = SEEDS_DIR / "axioms.json" if (SEEDS_DIR / "axioms.json").exists() else DATA_DIR / "axioms.json"
    KERNEL_PATH = SEEDS_DIR / "lek-1-kernel.txt" if (SEEDS_DIR / "lek-1-kernel.txt").exists() else DATA_DIR / "lek-1-kernel.txt"

    worker = args.worker or f"{socket.gethostname()}-{os.getpid()}"

    # ── Load token and check connectivity ─────────────────────────────
    token = get_influx_token()
    if not token:
        print("Error: no InfluxDB token available")
        sys.exit(1)

    # Test connectivity
    test = influx_query(token, "SELECT 1 AS ok")
    if not test:
        print(f"Error: cannot reach InfluxDB at {INFLUX_URL}")
        sys.exit(1)
    print(f"InfluxDB connected: {INFLUX_URL}")

    # ── Load prompts ──────────────────────────────────────────────────
    if not PROMPTS_PATH.exists():
        print(f"Error: prompts not found at {PROMPTS_PATH}")
        sys.exit(1)

    prompts = load_prompts()
    target = min(MAX_PROMPTS, len(prompts))
    print(f"Loaded {len(prompts)} prompts, targeting {target}")

    # ── Query completed from InfluxDB ─────────────────────────────────
    completed = get_completed_indices(token)
    remaining = [i for i in range(target) if i not in completed]
    print(f"Completed: {len(completed)} | Remaining: {len(remaining)}")

    if not remaining:
        print("All target prompts already completed!")
        return

    if args.dry_run:
        print(f"\n[DRY RUN] Would process {len(remaining)} prompts")
        print(f"  First 10: {remaining[:10]}")
        print(f"  Worker: {worker}")
        print(f"  Model: {args.model}")
        return

    # ── Setup output ──────────────────────────────────────────────────
    output_dir = DATA_DIR / "responses"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else output_dir / f"gold-{worker}.jsonl"
    print(f"Output: {output_path}")

    # ── Load model ────────────────────────────────────────────────────
    print(f"Loading model: {args.model}")
    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(args.model)
    sampler = make_sampler(temp=TEMPERATURE)
    print("Model loaded.")

    # ── Load axiom context ────────────────────────────────────────────
    system_text, kernel_text = load_axiom_context()
    print(f"Axiom context: {len(system_text)} + {len(kernel_text)} chars")

    # ── Generation loop ───────────────────────────────────────────────
    print(f"\nStarting generation as worker '{worker}'")
    print(f"{'='*60}")

    batch_start = time.time()
    generated = 0
    errors = 0
    limit = args.limit if args.limit > 0 else len(remaining)

    for idx in remaining:
        if generated >= limit:
            break

        seed = prompts[idx]

        try:
            response, elapsed = generate_response(
                model, tokenizer, sampler, system_text, kernel_text, seed["prompt"]
            )

            result = {
                "idx": idx,
                "seed_id": seed.get("seed_id", f"P_{idx:05d}"),
                "domain": seed.get("domain", "unknown"),
                "voice": seed.get("voice", "unknown"),
                "prompt": seed["prompt"],
                "response": response,
                "gen_time": round(elapsed, 1),
                "worker": worker,
            }

            # Write to local JSONL
            with open(output_path, "a") as f:
                f.write(json.dumps(result) + "\n")

            # Report to InfluxDB
            report_generation(token, worker, idx, seed, elapsed, len(response))

            generated += 1
            completed.add(idx)

            # Progress output
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

            # Periodically refresh completed set from InfluxDB
            # (picks up work done by other workers)
            if generated % REFRESH_EVERY == 0:
                new_completed = get_completed_indices(token)
                new_from_others = new_completed - completed
                if new_from_others:
                    print(f"  >> {len(new_from_others)} new completions from other workers")
                completed = new_completed
                remaining_now = [i for i in range(target) if i not in completed]
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

    # ── Final report ──────────────────────────────────────────────────
    elapsed_total = time.time() - batch_start
    report_stats(token, worker, len(completed), target)

    print(f"\n{'='*60}")
    print(f"Worker:    {worker}")
    print(f"Generated: {generated}")
    print(f"Errors:    {errors}")
    print(f"Total:     {len(completed)}/{target} ({len(completed)/target*100:.1f}%)")
    print(f"Time:      {elapsed_total/3600:.1f}h")
    print(f"Output:    {output_path}")


if __name__ == "__main__":
    main()
