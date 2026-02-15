#!/usr/bin/env python3
"""
Sync LEM repo model cards and benchmarks to HuggingFace.

Pushes README.md (model cards) from paper/hf-cards/ to each HuggingFace model repo,
and optionally syncs benchmark data to the lthn/LEK-benchmarks dataset.

Requirements:
  pip install huggingface_hub

Usage:
  python3 scripts/sync_hf.py                    # sync all model cards
  python3 scripts/sync_hf.py --models LEK-Gemma3-27B  # sync one model
  python3 scripts/sync_hf.py --benchmarks       # sync benchmark dataset
  python3 scripts/sync_hf.py --dry-run           # show what would be synced
  python3 scripts/sync_hf.py --all               # sync everything
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
CARDS_DIR = REPO_ROOT / "paper" / "hf-cards"
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
TRAINING_DIR = REPO_ROOT / "training"

HF_ORG = "lthn"

# Map card filename prefix to HF repo name
MODEL_MAP = {
    "LEK-Gemma3-1B-layered-v2": "LEK-Gemma3-1B-layered-v2",
    "LEK-Gemma3-1B-layered": "LEK-Gemma3-1B-layered",
    "LEK-Gemma3-4B": "LEK-Gemma3-4B",
    "LEK-Gemma3-12B": "LEK-Gemma3-12B",
    "LEK-Gemma3-27B": "LEK-Gemma3-27B",
    "LEK-GPT-OSS-20B": "LEK-GPT-OSS-20B",
    "LEK-Llama-3.1-8B": "LEK-Llama-3.1-8B",
    "LEK-Qwen-2.5-7B": "LEK-Qwen-2.5-7B",
    "LEK-Mistral-7B-v0.3": "LEK-Mistral-7B-v0.3",
}


def sync_model_cards(models=None, dry_run=False):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Error: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi()

    cards = sorted(CARDS_DIR.glob("*.md"))
    if not cards:
        print(f"No cards found in {CARDS_DIR}")
        return

    for card_path in cards:
        # Extract model name: LEK-Gemma3-12B-README.md → LEK-Gemma3-12B
        name = card_path.stem.replace("-README", "")
        if name not in MODEL_MAP:
            print(f"  Skip: {card_path.name} (not in MODEL_MAP)")
            continue

        if models and name not in models:
            continue

        repo_id = f"{HF_ORG}/{MODEL_MAP[name]}"

        if dry_run:
            print(f"  [DRY RUN] {card_path.name} → {repo_id}/README.md")
            continue

        try:
            api.upload_file(
                path_or_fileobj=str(card_path),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Update model card from LEM repo",
            )
            print(f"  Synced: {name} → {repo_id}")
        except Exception as e:
            print(f"  Error: {name} → {e}")


def sync_benchmarks(dry_run=False):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Error: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi()
    dataset_id = f"{HF_ORG}/LEK-benchmarks"

    # Collect benchmark files
    files = []
    for f in sorted(BENCHMARKS_DIR.rglob("*")):
        if f.is_file() and not f.name.startswith("."):
            rel = f.relative_to(REPO_ROOT)
            files.append((str(f), str(rel)))

    if dry_run:
        print(f"  [DRY RUN] Would upload {len(files)} files to {dataset_id}")
        for local, remote in files[:10]:
            print(f"    {remote}")
        if len(files) > 10:
            print(f"    ... and {len(files) - 10} more")
        return

    for local, remote in files:
        try:
            api.upload_file(
                path_or_fileobj=local,
                path_in_repo=remote,
                repo_id=dataset_id,
                repo_type="dataset",
                commit_message=f"Update benchmarks from LEM repo",
            )
        except Exception as e:
            print(f"  Error: {remote} → {e}")
    print(f"  Synced {len(files)} benchmark files to {dataset_id}")


def sync_training_parquet(dry_run=False):
    """Export training data as Parquet and sync to HuggingFace dataset."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from huggingface_hub import HfApi
    except ImportError:
        print("Error: pip install pyarrow huggingface_hub")
        sys.exit(1)

    import json

    api = HfApi()
    dataset_id = f"{HF_ORG}/LEK-training"
    output_dir = REPO_ROOT / "training" / "parquet"
    output_dir.mkdir(exist_ok=True)

    for split in ["train", "valid", "test"]:
        jsonl_path = TRAINING_DIR / f"{split}.jsonl"
        if not jsonl_path.exists():
            print(f"  Skip: {jsonl_path} not found")
            continue

        rows = []
        with open(jsonl_path) as f:
            for line in f:
                data = json.loads(line)
                msgs = data.get("messages", [])
                prompt = next((m["content"] for m in msgs if m["role"] == "user"), "")
                response = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
                rows.append({"prompt": prompt, "response": response, "messages": json.dumps(msgs)})

        table = pa.table({
            "prompt": [r["prompt"] for r in rows],
            "response": [r["response"] for r in rows],
            "messages": [r["messages"] for r in rows],
        })

        parquet_path = output_dir / f"{split}.parquet"
        pq.write_table(table, parquet_path)
        print(f"  Exported: {split}.parquet ({len(rows)} rows)")

        if dry_run:
            continue

        try:
            api.upload_file(
                path_or_fileobj=str(parquet_path),
                path_in_repo=f"data/{split}.parquet",
                repo_id=dataset_id,
                repo_type="dataset",
                commit_message=f"Update {split} split from LEM repo",
            )
            print(f"  Uploaded: {split}.parquet → {dataset_id}")
        except Exception as e:
            print(f"  Error uploading {split}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Sync LEM repo to HuggingFace")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Specific models to sync (default: all)")
    parser.add_argument("--benchmarks", action="store_true",
                        help="Sync benchmark dataset")
    parser.add_argument("--training", action="store_true",
                        help="Export training data as Parquet and sync")
    parser.add_argument("--all", action="store_true",
                        help="Sync everything (cards + benchmarks + training)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be synced")
    args = parser.parse_args()

    # Default to cards if nothing specified
    do_cards = args.all or (not args.benchmarks and not args.training)
    do_benchmarks = args.all or args.benchmarks
    do_training = args.all or args.training

    if do_cards:
        print("Syncing model cards...")
        sync_model_cards(models=args.models, dry_run=args.dry_run)

    if do_benchmarks:
        print("\nSyncing benchmarks...")
        sync_benchmarks(dry_run=args.dry_run)

    if do_training:
        print("\nExporting and syncing training data (Parquet)...")
        sync_training_parquet(dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
