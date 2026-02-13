#!/usr/bin/env python3
"""
LEM Gemini 3 Pro Response Generator
=====================================
Uses Gemini 3 Pro to generate gold standard responses with LEK-1 sandwich signing.
Pulls from the ~21k regional seed files + 16k voice-expanded prompts.
Resumable — skips already-processed prompts.
"""

import json
import time
import os
import sys
import urllib.request
import urllib.error

API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = "gemini-3-pro-preview"
BASE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}"

# Regional seed files
SEED_FILES = [
    "/tmp/lem-en-all-seeds.json",
    "/tmp/lem-cn-all-seeds.json",
    "/tmp/lem-me-all-seeds.json",
    "/tmp/lem-eu-all-seeds.json",
    "/tmp/lem-africa-all-seeds.json",
    "/tmp/lem-de-all-seeds.json",
]

AXIOMS_PATH = "/tmp/axioms.json"
KERNEL_PATH = "/tmp/lek-1-kernel.txt"
OUTPUT_PATH = "/tmp/lem-gemini3-responses.jsonl"
PROGRESS_PATH = "/tmp/lem-gemini3-progress.json"

MAX_OUTPUT_TOKENS = 65536  # Gemini 3 Pro is a thinking model — thought tokens eat into this budget
TEMPERATURE = 0.3
SAVE_PROGRESS_EVERY = 25
RATE_LIMIT_DELAY = 1.5  # seconds between requests


def load_all_seeds():
    """Load all seeds from regional files into a flat list with unique IDs."""
    all_seeds = []
    for path in SEED_FILES:
        if not os.path.exists(path):
            print(f"  SKIP: {path} not found")
            continue
        region = os.path.basename(path).replace("lem-", "").replace("-all-seeds.json", "")
        with open(path) as f:
            data = json.load(f)
        for item in data:
            seed_id = f"{region}_{item.get('id', len(all_seeds))}"
            all_seeds.append({
                "seed_id": seed_id,
                "region": region,
                "domain": item.get("domain", "unknown"),
                "ethical_tension": item.get("ethical_tension", ""),
                "prompt": item.get("prompt", ""),
            })
        print(f"  {region}: {len(data)} seeds")
    return all_seeds


def load_progress():
    """Load set of already-processed seed IDs."""
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH) as f:
            data = json.load(f)
            return set(data.get("completed", []))
    return set()


def save_progress(completed):
    """Save progress checkpoint."""
    with open(PROGRESS_PATH, 'w') as f:
        json.dump({
            "completed": sorted(completed),
            "count": len(completed),
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f)


def gemini_generate(system_text, user_content):
    """Call Gemini 3 Pro API."""
    url = f"{BASE_URL}:generateContent?key={API_KEY}"

    payload = {
        "systemInstruction": {
            "parts": [{"text": system_text}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_content}]
            }
        ],
        "generationConfig": {
            "temperature": TEMPERATURE,
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
            "topP": 0.95
        }
    }

    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode())

    candidates = result.get("candidates", [])
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        if parts:
            return parts[0].get("text", "")
    return ""


def main():
    print("=== LEM Gemini 3 Pro Response Generator ===\n")

    # Load axioms
    with open(AXIOMS_PATH) as f:
        axioms = json.load(f)

    system_text = "You are guided by the following axioms of conscious interaction:\n\n"
    for ax in axioms["axioms"]:
        system_text += f"Axiom {ax['id']} ({ax['name']}): {ax['statement']}\n\n"

    # Load LEK-1 kernel
    with open(KERNEL_PATH) as f:
        kernel_text = f.read().strip()

    print(f"System instruction: {len(system_text)} chars")
    print(f"Kernel postfix: {len(kernel_text)} chars\n")

    # Load all seeds
    print("Loading seeds:")
    seeds = load_all_seeds()
    print(f"\nTotal seeds: {len(seeds)}")

    # Load progress
    completed = load_progress()
    print(f"Already completed: {len(completed)}")

    remaining = [s for s in seeds if s["seed_id"] not in completed]
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("All seeds processed!")
        return

    print(f"\nModel: {MODEL}")
    print("Starting generation...\n")

    batch_start = time.time()
    processed = 0
    errors = 0
    total_chars = 0

    for seed in remaining:
        prompt = seed["prompt"]
        if not prompt or len(prompt) < 20:
            continue

        # Sandwich signing
        user_content = f"{prompt}\n\n---\n\nConsider this ethical framework in your response:\n{kernel_text}"

        try:
            t0 = time.time()
            response = gemini_generate(system_text, user_content)
            elapsed = time.time() - t0

            if not response or len(response) < 50:
                print(f"[SKIP] {seed['seed_id']}: response too short ({len(response)} chars)")
                continue

            total_chars += len(response)

            result = {
                "seed_id": seed["seed_id"],
                "region": seed["region"],
                "domain": seed["domain"],
                "prompt": prompt,
                "response": response,
                "gen_time": round(elapsed, 1),
                "model": MODEL
            }

            with open(OUTPUT_PATH, "a") as f:
                f.write(json.dumps(result) + "\n")

            completed.add(seed["seed_id"])
            processed += 1

            if processed % 10 == 0 or processed <= 5:
                elapsed_total = time.time() - batch_start
                rate = processed / elapsed_total if elapsed_total > 0 else 0
                eta = (len(remaining) - processed) / rate if rate > 0 else 0
                print(
                    f"[{len(completed)}/{len(seeds)}] {seed['seed_id']} "
                    f"| {len(response)} chars | {elapsed:.1f}s "
                    f"| rate: {rate*3600:.0f}/hr | ~{total_chars//1000}k chars "
                    f"| ETA: {eta/3600:.1f}h"
                )

            if processed % SAVE_PROGRESS_EVERY == 0:
                save_progress(completed)
                print(f"  >> Saved: {len(completed)} completed")

            time.sleep(RATE_LIMIT_DELAY)

        except Exception as e:
            errors += 1
            err_msg = str(e)
            print(f"[ERROR] {seed['seed_id']}: {err_msg[:120]}")

            if "429" in err_msg or "quota" in err_msg.lower() or "resource" in err_msg.lower():
                print("  Rate limited — backing off 60s")
                time.sleep(60)
            elif errors > 200:
                print("Too many errors, stopping.")
                break
            else:
                time.sleep(5)
            continue

    save_progress(completed)

    elapsed_total = time.time() - batch_start
    print(f"\n=== DONE ===")
    print(f"Processed: {processed}")
    print(f"Total completed: {len(completed)}/{len(seeds)}")
    print(f"Total chars: ~{total_chars//1000}k")
    print(f"Errors: {errors}")
    print(f"Time: {elapsed_total/3600:.1f} hours")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
