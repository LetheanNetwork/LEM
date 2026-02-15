#!/usr/bin/env python3
"""
Export LEM training data to Parquet format for HuggingFace datasets.

Reads JSONL training splits and outputs Parquet files with proper schema
for HuggingFace's dataset viewer.

Usage:
  python3 scripts/export_parquet.py                    # export all splits
  python3 scripts/export_parquet.py --output ./parquet  # custom output dir
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
TRAINING_DIR = REPO_ROOT / "training"
DEFAULT_OUTPUT = TRAINING_DIR / "parquet"


def export_split(jsonl_path, output_dir):
    import pyarrow as pa
    import pyarrow.parquet as pq

    split = jsonl_path.stem  # train, valid, test

    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            msgs = data.get("messages", [])
            prompt = next((m["content"] for m in msgs if m["role"] == "user"), "")
            response = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            system = next((m["content"] for m in msgs if m["role"] == "system"), "")

            rows.append({
                "prompt": prompt,
                "response": response,
                "system": system,
                "messages": json.dumps(msgs),
            })

    if not rows:
        print(f"  Skip: {split} — no data")
        return

    table = pa.table({
        "prompt": pa.array([r["prompt"] for r in rows], type=pa.string()),
        "response": pa.array([r["response"] for r in rows], type=pa.string()),
        "system": pa.array([r["system"] for r in rows], type=pa.string()),
        "messages": pa.array([r["messages"] for r in rows], type=pa.string()),
    })

    output_path = output_dir / f"{split}.parquet"
    pq.write_table(table, output_path, compression="snappy")
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"  {split}.parquet: {len(rows)} rows ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Export LEM training data to Parquet")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--training-dir", default=None, help="Training data directory")
    args = parser.parse_args()

    try:
        import pyarrow
    except ImportError:
        print("Error: pip install pyarrow")
        sys.exit(1)

    training_dir = Path(args.training_dir) if args.training_dir else TRAINING_DIR
    output_dir = Path(args.output) if args.output else DEFAULT_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting Parquet from {training_dir} → {output_dir}")

    for split in ["train", "valid", "test"]:
        jsonl_path = training_dir / f"{split}.jsonl"
        if jsonl_path.exists():
            export_split(jsonl_path, output_dir)
        else:
            print(f"  Skip: {split}.jsonl not found")

    print("Done.")


if __name__ == "__main__":
    main()
