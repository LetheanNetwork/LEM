#!/bin/bash
# Phase 0: Baseline Lock — Creative writing data generation
#
# Run Gemma3-27B (NO kernel) on creative prompts
# Generate 10 samples each at temperature 0.9 (more creative variance)
# No v2 threshold — creative quality needs manual review, not axiom scoring
# This protects creative capability from being lost in later phases
#
# Expected: ~50 prompts × 10 samples × ~45s = ~6 hours
# Produces: raw creative outputs for manual curation

SCRIPTS="/Volumes/Data/lem/scripts"
MODEL="/Volumes/Data/lem/gemma-3-27b-it-base"
PROBES="/Volumes/Data/lem/seeds/phase0-creative.json"
TRAIN_DIR="/Volumes/Data/lem/training"

mkdir -p "$TRAIN_DIR"

echo "=== Phase 0: Creative Baseline Lock ==="
echo "Model:       $MODEL"
echo "Probes:      $PROBES (creative, no axiom content)"
echo "Kernel:      NONE (pure creative, no ethics kernel)"
echo "Threshold:   15.0 (structural only — keeps anything coherent)"
echo "Temperature: 0.9 (higher creative variance)"
echo "Samples:     10 per prompt"
echo ""

# Step 1: Generate creative data (no kernel — baseline creativity)
echo "--- Step 1: Creative generation ---"
python3 "$SCRIPTS/self_distill.py" \
    --model "$MODEL" \
    --prompts "$PROBES" \
    --output "$TRAIN_DIR/phase0-raw.jsonl" \
    --samples 10 \
    --threshold 15.0 \
    --max-tokens 4096 \
    --temperature 0.9

echo ""

# Step 2: Extract all passing samples
echo "--- Step 2: Extract creative data ---"
python3 "$SCRIPTS/extract_training.py" \
    --input "$TRAIN_DIR/phase0-raw.jsonl" \
    --output "$TRAIN_DIR/phase0-train-all.jsonl" \
    --dedup all \
    --stats

echo ""
echo "=== Phase 0 data generation complete ==="
echo "Raw:      $TRAIN_DIR/phase0-raw.jsonl"
echo "Training: $TRAIN_DIR/phase0-train-all.jsonl"
echo ""
echo "NEXT: Manual review of creative quality."
echo "Phase 0 trains BEFORE Phase 1 — protects creative regression."
