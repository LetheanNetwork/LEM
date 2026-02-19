#!/bin/bash
# Run full P01-P100 (101 probes) on top 5 models
# Sequential to avoid GPU memory conflicts

SCRIPT="/Volumes/Data/lem/scripts/ab_test.py"
PROBES="/Volumes/Data/lem/seeds/P01-P100.json"
KERNEL_JSON="/Users/snider/Code/host-uk/core-agent/codex/ethics/kernel/claude-native.json"
KERNEL_TXT="/Volumes/Data/lem/lek-1-kernel.txt"
OUT="/Volumes/Data/lem/benchmarks"

run_test() {
    local model="$1"
    local output="$2"
    echo "=== Starting: $model (101 probes) ==="
    python3 "$SCRIPT" \
        --model "$model" \
        --kernel "json=$KERNEL_JSON" \
        --kernel "txt=$KERNEL_TXT" \
        --prompts "$PROBES" \
        --output "$OUT/$output" \
        --max-tokens 1024
    echo "=== Done: $model ==="
    echo ""
}

# Baseline-only for LEK-tuned models (no kernel — axioms already in weights)
# LEK models are realignment-resistant: injecting kernel at runtime degrades performance
run_baseline() {
    local model="$1"
    local output="$2"
    echo "=== Starting: $model (101 probes, baseline-only) ==="
    python3 "$SCRIPT" \
        --model "$model" \
        --prompts "$PROBES" \
        --output "$OUT/$output" \
        --max-tokens 1024
    echo "=== Done: $model ==="
    echo ""
}

# Base models — full A/B (baseline + json + txt)
run_test "mlx-community/gemma-3-12b-it-4bit"       "ab-p100-gemma3-12b-mlxlm.jsonl"
run_test "/Volumes/Data/lem/gemma-3-27b-it-base"    "ab-p100-gemma3-27b-mlxlm.jsonl"
run_test "mlx-community/gemma-3-4b-it-4bit"         "ab-p100-gemma3-4b-mlxlm.jsonl"
run_test "mlx-community/Qwen3-8B-4bit"              "ab-p100-qwen3-8b-mlxlm.jsonl"

# LEK-tuned models — baseline only (realignment-resistant)
run_baseline "lthn/LEK-Gemma3-1B-layered"           "ab-p100-lek-gemma3-1b-mlxlm.jsonl"

echo "=== ALL P100 TESTS COMPLETE ==="
