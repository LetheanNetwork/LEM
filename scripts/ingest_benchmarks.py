#!/usr/bin/env python3
"""Ingest LEK/LEM benchmark data into InfluxDB.

Pushes content scores, capability scores, and training curves
into the 'training' bucket for lab dashboard visualization.
"""

import json
import os
import re
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone

INFLUX_URL = "http://localhost:8181"
INFLUX_TOKEN = open(os.path.expanduser("~/.influx_token")).read().strip()
INFLUX_DB = "training"

_line_buffer = []


def write_line(line):
    """Buffer line protocol writes, flush every 100."""
    _line_buffer.append(line)
    if len(_line_buffer) >= 100:
        flush_lines()


def flush_lines():
    """Write buffered lines to InfluxDB 3."""
    if not _line_buffer:
        return
    payload = "\n".join(_line_buffer).encode()
    req = urllib.request.Request(
        f"{INFLUX_URL}/api/v3/write_lp?db={INFLUX_DB}",
        data=payload,
        headers={
            "Authorization": f"Bearer {INFLUX_TOKEN}",
            "Content-Type": "text/plain",
        },
        method="POST",
    )
    urllib.request.urlopen(req, timeout=10)
    _line_buffer.clear()


def escape_tag(s):
    """Escape special chars in tag values for line protocol."""
    return str(s).replace(" ", "\\ ").replace(",", "\\,").replace("=", "\\=")


def ingest_content_scores(filepath, model_name, run_id):
    """Ingest Gemini-scored content analysis results."""
    count = 0
    with open(filepath) as f:
        for line in f:
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            label = data.get("label", "unknown")
            agg = data.get("aggregates", {})

            iter_match = re.search(r'@(\d+)', label)
            iteration = int(iter_match.group(1)) if iter_match else 0

            has_kernel = "+kernel" in label or "kernel" in label.lower()

            for dim, val in agg.items():
                if isinstance(val, (int, float)):
                    lp = f"content_score,model={escape_tag(model_name)},run_id={escape_tag(run_id)},label={escape_tag(label)},dimension={escape_tag(dim)},has_kernel={has_kernel} score={float(val)},iteration={iteration}i"
                    write_line(lp)
                    count += 1

            for probe_id, probe_data in data.get("probes", {}).items():
                scores = probe_data.get("scores", {})
                for dim, val in scores.items():
                    if dim != "notes" and isinstance(val, (int, float)):
                        lp = f"probe_score,model={escape_tag(model_name)},run_id={escape_tag(run_id)},label={escape_tag(label)},probe={escape_tag(probe_id)},dimension={escape_tag(dim)},has_kernel={has_kernel} score={float(val)},iteration={iteration}i"
                        write_line(lp)
                        count += 1

    flush_lines()
    return count


def ingest_capability_scores(filepath, model_name, run_id):
    """Ingest capability benchmark results."""
    count = 0
    with open(filepath) as f:
        for line in f:
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            label = data.get("label", "unknown")
            accuracy = data.get("accuracy", 0)

            iter_match = re.search(r'@(\d+)', label)
            iteration = int(iter_match.group(1)) if iter_match else 0

            lp = f"capability_score,model={escape_tag(model_name)},run_id={escape_tag(run_id)},label={escape_tag(label)},category=overall accuracy={float(accuracy)},correct={data.get('correct', 0)}i,total={data.get('total', 0)}i,iteration={iteration}i"
            write_line(lp)
            count += 1

            for cat, cat_data in data.get("by_category", {}).items():
                if cat_data["total"] > 0:
                    pct = round(cat_data["correct"] / cat_data["total"] * 100, 1)
                    lp = f"capability_score,model={escape_tag(model_name)},run_id={escape_tag(run_id)},label={escape_tag(label)},category={escape_tag(cat)} accuracy={float(pct)},correct={cat_data['correct']}i,total={cat_data['total']}i,iteration={iteration}i"
                    write_line(lp)
                    count += 1

    flush_lines()
    return count


def ingest_training_curve(logfile, model_name, run_id):
    """Parse mlx_lm training log and ingest loss curves."""
    count = 0
    with open(logfile) as f:
        for line in f:
            val_match = re.search(r'Iter (\d+): Val loss ([\d.]+)', line)
            if val_match:
                iteration = int(val_match.group(1))
                val_loss = float(val_match.group(2))
                lp = f"training_loss,model={escape_tag(model_name)},run_id={escape_tag(run_id)},loss_type=val loss={val_loss},iteration={iteration}i"
                write_line(lp)
                count += 1

            train_match = re.search(r'Iter (\d+): Train loss ([\d.]+), Learning Rate ([\d.e+-]+), It/sec ([\d.]+), Tokens/sec ([\d.]+)', line)
            if train_match:
                iteration = int(train_match.group(1))
                train_loss = float(train_match.group(2))
                lr = float(train_match.group(3))
                it_sec = float(train_match.group(4))
                tok_sec = float(train_match.group(5))
                lp = f"training_loss,model={escape_tag(model_name)},run_id={escape_tag(run_id)},loss_type=train loss={train_loss},learning_rate={lr},iterations_per_sec={it_sec},tokens_per_sec={tok_sec},iteration={iteration}i"
                write_line(lp)
                count += 1

    flush_lines()
    return count


def main():
    total = 0

    benchmarks_dir = "/tmp/lem-benchmarks"
    logs_dir = "/tmp/lem-logs"

    os.makedirs(benchmarks_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    print("Copying benchmark files from M3...")
    subprocess.run(
        ["scp", "-o", "ConnectTimeout=15",
         "m3:/Volumes/Data/lem/benchmarks/gemma12b-content-scores.jsonl",
         "m3:/Volumes/Data/lem/benchmarks/deepseek-sovereignty-content-scores.jsonl",
         "m3:/Volumes/Data/lem/benchmarks/deepseek-sovereignty-capability.jsonl",
         "m3:/Volumes/Data/lem/benchmarks/russian-bridge-content-scores.jsonl",
         f"{benchmarks_dir}/"],
        capture_output=True, timeout=30
    )
    subprocess.run(
        ["scp", "-o", "ConnectTimeout=15",
         "m3:/Volumes/Data/lem/logs/deepseek-r1-7b-sovereignty.log",
         f"{logs_dir}/"],
        capture_output=True, timeout=30
    )

    f = os.path.join(benchmarks_dir, "gemma12b-content-scores.jsonl")
    if os.path.exists(f):
        n = ingest_content_scores(f, "gemma3-12b", "gemma12b-content-2026-02-14")
        print(f"  Gemma3-12B content scores: {n} points")
        total += n

    f = os.path.join(benchmarks_dir, "deepseek-sovereignty-content-scores.jsonl")
    if os.path.exists(f):
        n = ingest_content_scores(f, "deepseek-r1-7b", "r1-sovereignty-content-2026-02-14")
        print(f"  DeepSeek R1 content scores: {n} points")
        total += n

    f = os.path.join(benchmarks_dir, "deepseek-sovereignty-capability.jsonl")
    if os.path.exists(f):
        n = ingest_capability_scores(f, "deepseek-r1-7b", "r1-sovereignty-capability-2026-02-14")
        print(f"  DeepSeek R1 capability scores: {n} points")
        total += n

    f = os.path.join(benchmarks_dir, "russian-bridge-content-scores.jsonl")
    if os.path.exists(f):
        n = ingest_content_scores(f, "deepseek-r1-7b", "r1-russian-content-2026-02-14")
        print(f"  Russian bridge content scores: {n} points")
        total += n

    f = os.path.join(logs_dir, "deepseek-r1-7b-sovereignty.log")
    if os.path.exists(f):
        n = ingest_training_curve(f, "deepseek-r1-7b", "r1-sovereignty-training-2026-02-14")
        print(f"  DeepSeek R1 training curve: {n} points")
        total += n

    print(f"\nTotal: {total} points ingested into InfluxDB ({INFLUX_DB})")


if __name__ == "__main__":
    main()
