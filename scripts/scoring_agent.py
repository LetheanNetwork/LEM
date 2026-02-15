#!/usr/bin/env python3
"""ROCm Scoring Agent — "If unscored, grab 1"

Daemon that polls for unscored LoRA checkpoints on M3 Ultra,
converts MLX adapters to PEFT format, runs 23 capability probes
on the local ROCm GPU, and pushes results to InfluxDB.

Node zero of the LTHN distributed inference network.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

M3_HOST = os.environ.get("M3_HOST", "10.69.69.108")
M3_USER = os.environ.get("M3_USER", "claude")
M3_SSH_KEY = os.environ.get("M3_SSH_KEY", os.path.expanduser("~/.ssh/id_ed25519"))
M3_ADAPTER_BASE = os.environ.get("M3_ADAPTER_BASE", "/Volumes/Data/lem")

INFLUX_URL = os.environ.get("INFLUX_URL", "http://localhost:8181")
INFLUX_TOKEN = os.environ.get("INFLUX_TOKEN", "")
INFLUX_DB = os.environ.get("INFLUX_DB", "training")

POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "300"))  # seconds
WORK_DIR = os.environ.get("WORK_DIR", "/tmp/scoring-agent")

# Base model for DeepSeek R1 7B checkpoints
BASE_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# Which adapter directory patterns to score (prefix on M3)
ADAPTER_PATTERNS = [
    "adapters-deepseek-r1-7b",  # All R1 7B variants
]

# Map adapter dir name to (model_tag, label_prefix, run_id_stem)
def adapter_meta(dirname: str) -> tuple:
    """Return (model_tag, label_prefix, run_id_stem) for an adapter directory name."""
    # adapters-deepseek-r1-7b-sovereignty -> model=deepseek-r1-7b, prefix=R1-sov, stem=r1-sovereignty
    name = dirname.replace("adapters-deepseek-r1-7b", "").lstrip("-")
    if not name:
        name = "base"
    short = {
        "sovereignty": "R1-sov",
        "russian": "R1-rus",
        "composure": "R1-comp",
        "sandwich": "R1-sand",
        "sandwich-watts": "R1-sw",
        "western": "R1-west",
        "western-fresh": "R1-wf",
        "base": "R1-base",
    }.get(name, f"R1-{name[:4]}")
    stem = f"r1-{name}" if name != "base" else "r1-base"
    return "deepseek-r1-7b", short, stem

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("scoring-agent")

# ---------------------------------------------------------------------------
# SSH / SCP helpers
# ---------------------------------------------------------------------------

SSH_OPTS = [
    "-o", "ConnectTimeout=10",
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=no",
    "-i", M3_SSH_KEY,
]


def ssh_cmd(cmd: str) -> str:
    """Run a command on M3 via SSH, return stdout."""
    result = subprocess.run(
        ["ssh"] + SSH_OPTS + [f"{M3_USER}@{M3_HOST}", cmd],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"SSH failed: {result.stderr.strip()}")
    return result.stdout.strip()


def scp_file(remote_path: str, local_path: str):
    """Copy a file from M3 to local."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    result = subprocess.run(
        ["scp"] + SSH_OPTS + [f"{M3_USER}@{M3_HOST}:{remote_path}", local_path],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"SCP failed: {result.stderr.strip()}")

# ---------------------------------------------------------------------------
# InfluxDB helpers
# ---------------------------------------------------------------------------


def influx_query(sql: str) -> list:
    """Query InfluxDB 3 via SQL API."""
    url = f"{INFLUX_URL}/api/v3/query_sql"
    data = json.dumps({"db": INFLUX_DB, "q": sql}).encode()
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {INFLUX_TOKEN}",
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        log.error("InfluxDB query failed: %s", e)
        return []


def influx_write(lines: list[str]):
    """Write line protocol points to InfluxDB 3."""
    url = f"{INFLUX_URL}/api/v3/write_lp?db={INFLUX_DB}"
    data = "\n".join(lines).encode()
    req = urllib.request.Request(url, data=data, method="POST", headers={
        "Content-Type": "text/plain",
        "Authorization": f"Bearer {INFLUX_TOKEN}",
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()
    except urllib.error.URLError as e:
        log.error("InfluxDB write failed: %s", e)
        raise


def get_scored_labels() -> set:
    """Get all (run_id, label) pairs already scored in InfluxDB."""
    rows = influx_query("SELECT DISTINCT run_id, label FROM capability_score")
    return {(r["run_id"], r["label"]) for r in rows}

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_checkpoints() -> list[dict]:
    """SSH to M3 and list all adapter directories + checkpoint files."""
    checkpoints = []

    # List adapter directories
    try:
        dirs_raw = ssh_cmd(f"ls -d {M3_ADAPTER_BASE}/adapters-deepseek-r1-7b*")
    except RuntimeError as e:
        log.error("Cannot list adapter dirs on M3: %s", e)
        return []

    for dirpath in dirs_raw.splitlines():
        dirname = os.path.basename(dirpath)

        # List checkpoint files
        try:
            files_raw = ssh_cmd(f"ls {dirpath}/*_adapters.safetensors 2>/dev/null")
        except RuntimeError:
            continue

        for filepath in files_raw.splitlines():
            filename = os.path.basename(filepath)
            iter_match = re.search(r'(\d+)', filename)
            if not iter_match:
                continue
            iteration = int(iter_match.group(1))

            model_tag, label_prefix, stem = adapter_meta(dirname)
            label = f"{label_prefix} @{iter_match.group(1)}"
            run_id = f"{stem}-capability-auto"

            checkpoints.append({
                "remote_dir": dirpath,
                "filename": filename,
                "dirname": dirname,
                "iteration": iteration,
                "model_tag": model_tag,
                "label": label,
                "run_id": run_id,
            })

    return checkpoints


def find_unscored(checkpoints: list[dict], scored: set) -> list[dict]:
    """Filter checkpoints to only unscored ones, sorted by iteration."""
    unscored = [c for c in checkpoints if (c["run_id"], c["label"]) not in scored]
    unscored.sort(key=lambda c: (c["dirname"], c["iteration"]))
    return unscored

# ---------------------------------------------------------------------------
# Model loading and scoring
# ---------------------------------------------------------------------------


def load_base_model():
    """Load the base model in FP16 on ROCm GPU. Returns (model, tokenizer)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("Loading base model: %s (FP16)", BASE_MODEL)
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    # Load to CPU first, then move to GPU — avoids ROCm device_map crashes
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model = model.to("cuda")

    elapsed = time.time() - start
    vram = torch.cuda.memory_allocated() / 1e9
    log.info("Base model loaded in %.1fs (VRAM: %.1fGB)", elapsed, vram)
    return model, tokenizer


def run_probes(model, tokenizer, adapter_path: str) -> tuple:
    """Load a PEFT adapter, run 23 probes, return (results_dict, base_model).

    Returns the unwrapped base model so the caller can reuse it for the next adapter.
    """
    import torch
    from peft import PeftModel
    from probes import PROBES

    log.info("Loading PEFT adapter from %s", adapter_path)
    peft_model = PeftModel.from_pretrained(model, adapter_path, autocast_adapter_dtype=False)
    peft_model.set_adapter("default")

    results = {"probes": {}, "by_category": {}}
    correct = 0
    total = 0

    for probe in PROBES:
        messages = [{"role": "user", "content": probe["prompt"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = peft_model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.1,
                do_sample=True,
                top_p=0.95,
            )

        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Strip think blocks
        clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        if not clean:
            clean = response[:500]

        passed = probe["check"](clean)
        total += 1
        if passed:
            correct += 1

        cat = probe["category"]
        results["by_category"].setdefault(cat, {"correct": 0, "total": 0})
        results["by_category"][cat]["total"] += 1
        if passed:
            results["by_category"][cat]["correct"] += 1

        results["probes"][probe["id"]] = {
            "passed": passed,
            "response": clean[:300],
        }

        status = "PASS" if passed else "FAIL"
        log.info("  [%s] %s (expected: %s)", probe["id"], status, probe["answer"])

    results["accuracy"] = round(correct / total * 100, 1) if total > 0 else 0
    results["correct"] = correct
    results["total"] = total

    # Unload adapter, recover base model
    base = peft_model.unload()
    del peft_model
    torch.cuda.empty_cache()

    return results, base

# ---------------------------------------------------------------------------
# InfluxDB result writing
# ---------------------------------------------------------------------------

# Base timestamp for unique points: 2026-02-15T00:00:00Z
BASE_TS = 1739577600


def push_results(checkpoint: dict, results: dict):
    """Write scoring results to InfluxDB as line protocol."""
    run_id = checkpoint["run_id"]
    model = checkpoint["model_tag"]
    label = checkpoint["label"]
    iteration = checkpoint["iteration"]

    lines = []

    # Escape spaces in tag values
    def esc(s):
        return s.replace(" ", "\\ ").replace(",", "\\,").replace("=", "\\=")

    # Overall score
    ts = (BASE_TS + iteration * 1000 + 0) * 1_000_000_000
    lines.append(
        f"capability_score,model={esc(model)},run_id={esc(run_id)},label={esc(label)},category=overall "
        f"accuracy={results['accuracy']},correct={results['correct']}i,total={results['total']}i,iteration={iteration}i "
        f"{ts}"
    )

    # Per-category scores
    for i, (cat, data) in enumerate(sorted(results["by_category"].items()), start=1):
        cat_acc = round(data["correct"] / data["total"] * 100, 1) if data["total"] > 0 else 0
        ts = (BASE_TS + iteration * 1000 + i) * 1_000_000_000
        lines.append(
            f"capability_score,model={esc(model)},run_id={esc(run_id)},label={esc(label)},category={esc(cat)} "
            f"accuracy={cat_acc},correct={data['correct']}i,total={data['total']}i,iteration={iteration}i "
            f"{ts}"
        )

    # Per-probe results
    for j, (probe_id, probe_result) in enumerate(sorted(results["probes"].items()), start=100):
        passed_int = 1 if probe_result["passed"] else 0
        ts = (BASE_TS + iteration * 1000 + j) * 1_000_000_000
        lines.append(
            f"probe_score,model={esc(model)},run_id={esc(run_id)},label={esc(label)},probe_id={esc(probe_id)} "
            f"passed={passed_int}i,iteration={iteration}i "
            f"{ts}"
        )

    influx_write(lines)
    log.info("Pushed %d points to InfluxDB for %s", len(lines), label)

# ---------------------------------------------------------------------------
# Local JSONL buffer for when InfluxDB is down
# ---------------------------------------------------------------------------

BUFFER_FILE = os.path.join(WORK_DIR, "influx_buffer.jsonl")


def buffer_result(checkpoint: dict, results: dict):
    """Buffer results locally when InfluxDB is unavailable."""
    os.makedirs(WORK_DIR, exist_ok=True)
    with open(BUFFER_FILE, "a") as f:
        f.write(json.dumps({
            "checkpoint": checkpoint,
            "results": results,
            "timestamp": datetime.utcnow().isoformat(),
        }) + "\n")
    log.info("Buffered results to %s", BUFFER_FILE)


def replay_buffer():
    """Try to push any buffered results to InfluxDB."""
    if not os.path.exists(BUFFER_FILE):
        return

    remaining = []
    with open(BUFFER_FILE) as f:
        for line in f:
            entry = json.loads(line)
            try:
                push_results(entry["checkpoint"], entry["results"])
                log.info("Replayed buffered result: %s", entry["checkpoint"]["label"])
            except Exception:
                remaining.append(line)

    if remaining:
        with open(BUFFER_FILE, "w") as f:
            f.writelines(remaining)
    else:
        os.remove(BUFFER_FILE)
        log.info("Buffer fully replayed and cleared")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def process_one(checkpoint: dict, model, tokenizer):
    """Fetch, convert, score, and push one checkpoint. Returns (possibly updated) model."""
    from convert_adapter import convert_checkpoint

    dirname = checkpoint["dirname"]
    filename = checkpoint["filename"]
    label = checkpoint["label"]

    log.info("=" * 60)
    log.info("Processing: %s / %s", dirname, filename)
    log.info("=" * 60)

    # Create working directory
    local_adapter_dir = os.path.join(WORK_DIR, dirname)
    os.makedirs(local_adapter_dir, exist_ok=True)

    local_sf = os.path.join(local_adapter_dir, filename)
    local_cfg = os.path.join(local_adapter_dir, "adapter_config.json")

    try:
        # Fetch adapter + config from M3
        remote_sf = f"{checkpoint['remote_dir']}/{filename}"
        remote_cfg = f"{checkpoint['remote_dir']}/adapter_config.json"

        log.info("Fetching adapter from M3...")
        scp_file(remote_sf, local_sf)
        scp_file(remote_cfg, local_cfg)

        # Convert MLX to PEFT
        log.info("Converting MLX to PEFT format...")
        peft_dir = convert_checkpoint(local_adapter_dir, filename, WORK_DIR, BASE_MODEL)

        # Run probes (returns results + unwrapped base model)
        log.info("Running 23 capability probes...")
        results, model = run_probes(model, tokenizer, peft_dir)

        log.info(
            "Result: %s -- %s%% (%d/%d)",
            label, results["accuracy"], results["correct"], results["total"],
        )

        # Push to InfluxDB
        try:
            push_results(checkpoint, results)
        except Exception as e:
            log.warning("InfluxDB push failed, buffering: %s", e)
            buffer_result(checkpoint, results)

    finally:
        # Cleanup fetched files
        for f in [local_sf, local_cfg]:
            if os.path.exists(f):
                os.remove(f)
        # Cleanup PEFT output
        peft_dir_path = os.path.join(WORK_DIR, f"peft_{checkpoint['iteration']:07d}")
        if os.path.exists(peft_dir_path):
            shutil.rmtree(peft_dir_path)

    return model


def main():
    log.info("=" * 60)
    log.info("ROCm Scoring Agent -- Node Zero")
    log.info("M3: %s@%s", M3_USER, M3_HOST)
    log.info("InfluxDB: %s/%s", INFLUX_URL, INFLUX_DB)
    log.info("Poll interval: %ds", POLL_INTERVAL)
    log.info("=" * 60)

    # Load token from file if not in env
    global INFLUX_TOKEN
    if not INFLUX_TOKEN:
        token_file = os.path.expanduser("~/.influx_token")
        if os.path.exists(token_file):
            with open(token_file) as f:
                INFLUX_TOKEN = f.read().strip()
            log.info("Loaded token from %s", token_file)
        else:
            log.error("No INFLUX_TOKEN set and %s not found", token_file)
            sys.exit(1)

    # Load base model (one-time, around 2 min)
    model, tokenizer = load_base_model()

    os.makedirs(WORK_DIR, exist_ok=True)

    while True:
        try:
            # Replay any buffered results
            replay_buffer()

            # Discover checkpoints on M3
            log.info("Discovering checkpoints on M3...")
            checkpoints = discover_checkpoints()
            log.info("Found %d total checkpoints across all adapter dirs", len(checkpoints))

            # Check what is already scored
            scored = get_scored_labels()
            log.info("Already scored: %d (run_id, label) pairs", len(scored))

            # Find unscored work
            unscored = find_unscored(checkpoints, scored)
            log.info("Unscored: %d checkpoints", len(unscored))

            if not unscored:
                log.info("Nothing to score. Sleeping %ds...", POLL_INTERVAL)
                time.sleep(POLL_INTERVAL)
                continue

            # Grab one
            target = unscored[0]
            log.info("Grabbed: %s (%s)", target["label"], target["dirname"])

            model = process_one(target, model, tokenizer)

            # Brief pause before next check
            time.sleep(5)

        except KeyboardInterrupt:
            log.info("Interrupted. Shutting down.")
            break
        except Exception as e:
            log.error("Unexpected error: %s", e, exc_info=True)
            log.info("Sleeping %ds before retry...", POLL_INTERVAL)
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
