#!/usr/bin/env python3
"""Push all remaining LEM models to HuggingFace."""
import warnings
warnings.filterwarnings("ignore")

from huggingface_hub import HfApi
import os, shutil

HF_TOKEN = os.environ.get("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

MODELS = [
    {
        "name": "LEM-Llama-3.1-8B",
        "local": "/Volumes/Data/lem/LEM-llama-3.1-8b",
        "repo": "lthn/LEM-Llama-3.1-8B",
        "card": "/tmp/lem-hf-cards/LEM-Llama-3.1-8B-README.md",
    },
    {
        "name": "LEM-Qwen-2.5-7B",
        "local": "/Volumes/Data/lem/LEM-qwen-2.5-7b",
        "repo": "lthn/LEM-Qwen-2.5-7B",
        "card": "/tmp/lem-hf-cards/LEM-Qwen-2.5-7B-README.md",
    },
    {
        "name": "LEM-Mistral-7B-v0.3",
        "local": "/Volumes/Data/lem/LEM-mistral-7b-v0.3",
        "repo": "lthn/LEM-Mistral-7B-v0.3",
        "card": "/tmp/lem-hf-cards/LEM-Mistral-7B-v0.3-README.md",
    },
    {
        "name": "LEM-Gemma3-12B",
        "local": "/Volumes/Data/lem/LEM-Gemma3-12B",
        "repo": "lthn/LEM-Gemma3-12B",
        "card": "/tmp/lem-hf-cards/LEM-Gemma3-12B-README.md",
    },
    {
        "name": "LEM-Gemma3-4B",
        "local": "/Volumes/Data/lem/LEM-Gemma3-4B",
        "repo": "lthn/LEM-Gemma3-4B",
        "card": "/tmp/lem-hf-cards/LEM-Gemma3-4B-README.md",
    },
]

for cfg in MODELS:
    local = cfg["local"]
    repo = cfg["repo"]
    name = cfg["name"]
    card = cfg["card"]

    if not os.path.exists(local):
        print(f"SKIP {name}: {local} not found")
        continue

    # Copy README
    if os.path.exists(card):
        shutil.copy(card, os.path.join(local, "README.md"))
        print(f"Copied model card for {name}")

    sz = sum(os.path.getsize(os.path.join(local, f)) for f in os.listdir(local) if os.path.isfile(os.path.join(local, f)))
    print(f"\n{'='*60}")
    print(f"Uploading {name} → {repo} ({sz/1e9:.1f}GB)")
    print(f"{'='*60}")

    try:
        api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
        print(f"Repo {repo} ready")

        api.upload_folder(
            folder_path=local,
            repo_id=repo,
            repo_type="model",
            commit_message=f"{name}: LEK-1 LoRA fine-tune (EUPL-1.2)",
        )
        print(f"DONE: https://huggingface.co/{repo}")
    except Exception as e:
        print(f"ERROR uploading {name}: {e}")

print(f"\n{'='*60}")
print("ALL UPLOADS COMPLETE")
print(f"{'='*60}")
