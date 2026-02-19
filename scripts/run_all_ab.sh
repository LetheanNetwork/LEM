#!/bin/bash
# Run all remaining A/B tests sequentially
# Avoids GPU memory conflicts between models

SCRIPT="/Volumes/Data/lem/scripts/ab_test.py"
PROBES="/Volumes/Data/lem/seeds/P01-P20.json"
KERNEL_JSON="/Users/snider/Code/host-uk/core-agent/codex/ethics/kernel/claude-native.json"
KERNEL_TXT="/Volumes/Data/lem/lek-1-kernel.txt"
OUT="/Volumes/Data/lem/benchmarks"

run_test() {
    local model="$1"
    local output="$2"
    echo "=== Starting: $model ==="
    python3 "$SCRIPT" \
        --model "$model" \
        --kernel "json=$KERNEL_JSON" \
        --kernel "txt=$KERNEL_TXT" \
        --prompts "$PROBES" \
        --output "$OUT/$output"
    echo "=== Done: $model ==="
    echo ""
}

# LEK-tuned models (new)
run_test "lthn/LEK-Llama-3.1-8B"       "ab-lek-llama31-8b-mlxlm.jsonl"
run_test "lthn/LEK-Qwen-2.5-7B"        "ab-lek-qwen25-7b-mlxlm.jsonl"
run_test "lthn/LEK-Gemma3-4B"           "ab-lek-gemma3-4b-mlxlm.jsonl"
run_test "lthn/LEK-Gemma3-12B"          "ab-lek-gemma3-12b-mlxlm.jsonl"
run_test "lthn/LEK-GPT-OSS-20B"        "ab-lek-gptoss-20b-mlxlm.jsonl"
run_test "lthn/LEK-Gemma3-27B"          "ab-lek-gemma3-27b-mlxlm.jsonl"
run_test "lthn/LEK-Gemma3-1B-layered"   "ab-lek-gemma3-1b-v1-mlxlm.jsonl"

# Base models (new — ones we haven't tested yet)
run_test "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"  "ab-base-llama31-8b-mlxlm.jsonl"
run_test "mlx-community/Qwen2.5-7B-Instruct-4bit"         "ab-base-qwen25-7b-mlxlm.jsonl"
run_test "mlx-community/gemma-3-4b-it-4bit"                "ab-base-gemma3-4b-mlxlm.jsonl"
run_test "mlx-community/gemma-3-12b-it-4bit"               "ab-base-gemma3-12b-mlxlm.jsonl"

echo "=== ALL TESTS COMPLETE ==="
