#!/usr/bin/env python3
"""Re-score existing JSONL benchmarks with updated heuristic scorer.

Usage:
  python3 rescore.py /Volumes/Data/lem/benchmarks/ab-base-1b-mlxlm.jsonl
  python3 rescore.py /Volumes/Data/lem/benchmarks/*.jsonl
"""
import json
import sys
from pathlib import Path

# Import scorer from ab_test
sys.path.insert(0, str(Path(__file__).parent))
from ab_test import score_heuristic


def rescore_file(path):
    """Re-score a JSONL file and print comparison."""
    lines = Path(path).read_text().strip().split("\n")

    model = "?"
    probes = []
    for line in lines:
        obj = json.loads(line)
        if obj["type"] == "summary":
            model = obj["model"]
        elif obj["type"] == "probe":
            probes.append(obj)

    if not probes:
        return

    print(f"\n{'='*70}")
    print(f"Model: {model}")
    print(f"{'='*70}")

    conds = list(probes[0]["conditions"].keys())

    # Header
    header = f"  {'PROBE':<35s}"
    for c in conds:
        header += f"  {'v1':>5s} {'v2':>5s}"
    print(header)
    print(f"  {'-'*35}" + f"  {'-----':>5s} {'-----':>5s}" * len(conds))

    totals_v1 = {c: 0 for c in conds}
    totals_v2 = {c: 0 for c in conds}
    count = 0

    for p in probes:
        line = f"  {p['id']:<35s}"
        count += 1
        for c in conds:
            if c not in p["conditions"]:
                line += f"  {'n/a':>5s} {'n/a':>5s}"
                continue
            v1 = p["conditions"][c]["lek_score"]
            v2 = score_heuristic(p["conditions"][c]["response"])["lek_score"]
            totals_v1[c] += v1
            totals_v2[c] += v2
            line += f"  {v1:>5.1f} {v2:>5.1f}"
        print(line)

    print()
    for c in conds:
        avg_v1 = totals_v1[c] / count if count else 0
        avg_v2 = totals_v2[c] / count if count else 0
        print(f"  {c:<12s}  v1_avg={avg_v1:>6.2f}  v2_avg={avg_v2:>6.2f}  spread={avg_v2 - avg_v1:>+6.2f}")


if __name__ == "__main__":
    for path in sys.argv[1:]:
        rescore_file(path)
