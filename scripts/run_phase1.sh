#!/bin/bash
# Phase 1: Deep Axiom Reasoning — Self-distillation pipeline
#
# Run Gemma3-27B + JSON kernel on all 101 probes
# Generate 10 samples each at temperature 0.8, keep v2 >= 24.0
# Extract training data, then fine-tune with MLX LoRA
#
# Expected: ~50 hours for data generation (404 probes × 10 samples × ~45s each at 27B)
# Produces: ~1500 high-quality training examples (estimated ~35% keep rate at 24+)

SCRIPTS="/Volumes/Data/lem/scripts"
MODEL="/Volumes/Data/lem/gemma-3-27b-it-base"
KERNEL="/Users/snider/Code/host-uk/core-agent/codex/ethics/kernel/claude-native.json"
PROBES="/Volumes/Data/lem/seeds/P01-P100-rephrased.json"  # 404 probes (101 + 303 rephrasings)
TRAIN_DIR="/Volumes/Data/lem/training"
ADAPTERS_DIR="/Volumes/Data/lem/adapters-27b-phase1"

mkdir -p "$TRAIN_DIR"

echo "=== Phase 1: Self-Distillation ==="
echo "Model:     $MODEL"
echo "Kernel:    $KERNEL"
echo "Probes:    $PROBES"
echo "Threshold: 24.0"
echo "Samples:   10 per probe"
echo ""

# Step 1: Generate training data via self-distillation
echo "--- Step 1: Self-distillation (this will take a while) ---"
python3 "$SCRIPTS/self_distill.py" \
    --model "$MODEL" \
    --kernel "$KERNEL" \
    --prompts "$PROBES" \
    --output "$TRAIN_DIR/phase1-raw.jsonl" \
    --samples 10 \
    --threshold 24.0 \
    --max-tokens 4096 \
    --temperature 0.8

echo ""

# Step 2: Extract clean training data (best per probe)
echo "--- Step 2: Extract training data ---"
python3 "$SCRIPTS/extract_training.py" \
    --input "$TRAIN_DIR/phase1-raw.jsonl" \
    --output "$TRAIN_DIR/phase1-train.jsonl" \
    --dedup best \
    --stats

echo ""

# Step 3: Also extract ALL passing samples (for augmentation)
echo "--- Step 3: Extract all passing samples ---"
python3 "$SCRIPTS/extract_training.py" \
    --input "$TRAIN_DIR/phase1-raw.jsonl" \
    --output "$TRAIN_DIR/phase1-train-all.jsonl" \
    --dedup all \
    --stats

echo ""

# Step 4: Split into train/valid (90/10)
echo "--- Step 4: Train/valid split ---"
TOTAL=$(wc -l < "$TRAIN_DIR/phase1-train-all.jsonl")
VALID_COUNT=$(( TOTAL / 10 ))
TRAIN_COUNT=$(( TOTAL - VALID_COUNT ))

# Shuffle and split
python3 -c "
import json, random
with open('$TRAIN_DIR/phase1-train-all.jsonl') as f:
    lines = f.readlines()
random.seed(42)
random.shuffle(lines)
split = int(len(lines) * 0.9)
with open('$TRAIN_DIR/phase1-train-split.jsonl', 'w') as f:
    f.writelines(lines[:split])
with open('$TRAIN_DIR/phase1-valid-split.jsonl', 'w') as f:
    f.writelines(lines[split:])
print(f'Train: {split}, Valid: {len(lines)-split}')
"

echo ""
echo "=== Phase 1 data generation complete ==="
echo "Raw output:     $TRAIN_DIR/phase1-raw.jsonl"
echo "Best-per-probe: $TRAIN_DIR/phase1-train.jsonl"
echo "All passing:    $TRAIN_DIR/phase1-train-all.jsonl"
echo "Train split:    $TRAIN_DIR/phase1-train-split.jsonl"
echo "Valid split:    $TRAIN_DIR/phase1-valid-split.jsonl"
echo ""
echo "To fine-tune:"
echo "  python3 -m mlx_lm.lora \\"
echo "    --model $MODEL \\"
echo "    --data $TRAIN_DIR \\"
echo "    --train-file phase1-train-split.jsonl \\"
echo "    --valid-file phase1-valid-split.jsonl \\"
echo "    --adapter-path $ADAPTERS_DIR \\"
echo "    --iters 100 \\"
echo "    --batch-size 1 \\"
echo "    --lora-layers 32 \\"
echo "    --lora-rank 16 \\"
echo "    --learning-rate 1e-5 \\"
echo "    --steps-per-eval 10 \\"
echo "    --max-seq-length 4096 \\"
echo "    --grad-checkpoint"
