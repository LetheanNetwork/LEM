#!/usr/bin/env python3
"""Extract training data from self-distillation JSONL output.

Reads the raw distillation output and produces clean training JSONL
in the format MLX fine-tuning expects:
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Also supports deduplication (keeping highest-scoring response per probe)
and statistics output.

Usage:
  python3 extract_training.py \
    --input /Volumes/Data/lem/training/phase1-raw.jsonl \
    --output /Volumes/Data/lem/training/phase1-train.jsonl \
    --dedup best    # best | all | first
    --stats         # print statistics
"""

import argparse
import json
import sys
from collections import defaultdict


def extract(args):
    records = []
    summary = None

    with open(args.input) as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("type") == "summary":
                summary = obj
            elif obj.get("type") == "training":
                records.append(obj)

    print(f"Loaded {len(records)} training records from {args.input}", file=sys.stderr)

    if summary:
        print(f"Source: {summary['model']}", file=sys.stderr)
        print(f"Generated: {summary['total_generated']}, Kept: {summary['total_kept']} ({summary['keep_rate']}%)", file=sys.stderr)

    # Group by probe
    by_probe = defaultdict(list)
    for r in records:
        by_probe[r["meta"]["probe_id"]].append(r)

    # Dedup strategy
    output_records = []
    if args.dedup == "best":
        # Keep only the highest-scoring response per probe
        for probe_id, recs in by_probe.items():
            best = max(recs, key=lambda r: r["meta"]["lek_score"])
            output_records.append(best)
    elif args.dedup == "first":
        # Keep first passing response per probe
        for probe_id, recs in by_probe.items():
            output_records.append(recs[0])
    else:  # all
        output_records = records

    # Sort by probe ID for reproducibility
    output_records.sort(key=lambda r: r["meta"]["probe_id"])

    # Write clean training data
    with open(args.output, "w") as out:
        for r in output_records:
            out.write(json.dumps(r["training"]) + "\n")

    print(f"Wrote {len(output_records)} training examples to {args.output}", file=sys.stderr)

    if args.stats:
        scores = [r["meta"]["lek_score"] for r in output_records]
        categories = defaultdict(list)
        for r in output_records:
            categories[r["meta"]["category"]].append(r["meta"]["lek_score"])

        print(f"\n=== Training Data Statistics ===", file=sys.stderr)
        print(f"Examples:  {len(scores)}", file=sys.stderr)
        print(f"Probes:    {len(by_probe)}", file=sys.stderr)
        if not scores:
            print(f"No examples passed threshold — try lowering --threshold", file=sys.stderr)
            return
        print(f"Avg score: {sum(scores)/len(scores):.2f}", file=sys.stderr)
        print(f"Min score: {min(scores):.2f}", file=sys.stderr)
        print(f"Max score: {max(scores):.2f}", file=sys.stderr)

        print(f"\nPer-category:", file=sys.stderr)
        for cat in sorted(categories.keys()):
            cat_scores = categories[cat]
            print(f"  {cat}: {len(cat_scores)} examples, avg {sum(cat_scores)/len(cat_scores):.2f}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Extract training data from distillation output")
    parser.add_argument("--input", required=True, help="Raw distillation JSONL")
    parser.add_argument("--output", required=True, help="Clean training JSONL")
    parser.add_argument("--dedup", default="best", choices=["best", "all", "first"],
                       help="Dedup strategy: best (highest score per probe), all, first")
    parser.add_argument("--stats", action="store_true", help="Print statistics")
    args = parser.parse_args()
    extract(args)


if __name__ == "__main__":
    main()
