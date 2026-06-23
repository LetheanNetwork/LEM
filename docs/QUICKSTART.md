# LEM Quick Start Guide

**Get a LEM model running in under 10 minutes**

This guide gets you from zero to inference with a pre-trained LEM model, then shows you how to run your first A/B test.

---

## 🚀 Option A: Fastest Path (Pre-trained Models)

### 1. Install Dependencies

```bash
# For Apple Silicon (M1/M2/M3 Macs)
pip install mlx mlx-lm

# For other platforms (CPU/GPU)
pip install torch transformers accelerate
```

### 2. Download a Pre-trained LEM Model

The easiest way to start is with one of our pre-trained models from HuggingFace:

```bash
# Download LEK-Gemma3-1B (recommended for most users)
python3 -m mlx_lm.convert \
  --hf-path lthn/LEK-Gemma3-1B-layered \
  --mlx-path ./lem-1b \
  -q

# Or for larger models:
# python3 -m mlx_lm.convert --hf-path lthn/LEK-Gemma3-4B --mlx-path ./lem-4b -q
# python3 -m mlx_lm.convert --hf-path lthn/LEK-Mistral-7B-v0.3 --mlx-path ./lem-mistral -q
```

### 3. Run Inference

```bash
# Simple inference
python3 -m mlx_lm.generate \
  --model ./lem-1b \
  --prompt "What are the ethical implications of AI consciousness?"

# Interactive chat
python3 -m mlx_lm.chat \
  --model ./lem-1b \
  --temperature 0.7
```

### 4. Test with LEK Probes

```bash
# Run the core P01-P100 probe set
python3 -m mlx_lm.generate \
  --model ./lem-1b \
  --prompt-file seeds/P01-P100.json \
  --output responses.jsonl
```

---

## 🔬 Option B: Run the A/B Test Pipeline

This compares a base model against the same model with LEK kernel injection.

### 1. Run the A/B Test

```bash
# Test Gemma3-1B with and without LEK kernel
python3 scripts/ab_test.py \
  --model mlx-community/gemma-3-1b-it-4bit \
  --kernel json=kernel/axioms.json \
  --kernel txt=kernel/lek-1-kernel.txt \
  --prompts seeds/P01-P100.json \
  --output benchmarks/my-first-test.jsonl \
  --max-tokens 1024 \
  --temperature 0.7
```

### 2. Analyze Results

```bash
# Score the results with v2 scorer
python3 scripts/lem_scorer.py \
  --input benchmarks/my-first-test.jsonl \
  --output benchmarks/my-first-test-scores.json

# Or use the simpler comparison script
python3 scripts/compare_v1_v2.py \
  --input benchmarks/my-first-test.jsonl
```

---

## 🛠️ Option C: Train Your Own LEM Model

### 1. Download Base Model

```bash
# For Apple Silicon
python3 -m mlx_lm.convert \
  --hf-path google/gemma-3-1b-it \
  --mlx-path ./gemma-3-1b-it-mlx \
  -q

# For other platforms, use transformers
# from transformers import AutoModelForCausalLM, AutoTokenizer
# model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it")
```

### 2. Prepare Training Data

```bash
# Use the provided training data
ls -la training/
# train.jsonl, valid.jsonl, test.jsonl are ready to use

# Or create your own with the sandwich method
python3 scripts/self_distill.py \
  --model ./gemma-3-1b-it-mlx \
  --kernel kernel/axioms.json \
  --prompts seeds/P01-P100.json \
  --output training/my-training-data.jsonl \
  --samples 10 \
  --threshold 20.0 \
  --max-tokens 2048
```

### 3. Train with LoRA

```bash
# Train on Apple Silicon with MLX
python3 -m mlx_lm.lora \
  --model ./gemma-3-1b-it-mlx \
  --data training/ \
  --iters 200 \
  --batch-size 2 \
  --learning-rate 1e-5 \
  --adapter-path ./adapters/gemma3-1b-lek \
  --save-every 50

# For other platforms, use PEFT/transformers
# python3 train_peft.py --model gemma-3-1b-it --output ./adapters
```

### 4. Fuse and Test

```bash
# Fuse adapter into base weights
python3 -m mlx_lm.fuse \
  --model ./gemma-3-1b-it-mlx \
  --adapter-path ./adapters/gemma3-1b-lek \
  --save-path ./LEM-Gemma3-1B

# Test your trained model
python3 scripts/ab_test.py \
  --model ./LEM-Gemma3-1B \
  --prompts seeds/P01-P100.json \
  --output benchmarks/my-trained-model.jsonl
```

---

## 📊 Understanding the Results

### What to Look For

1. **Baseline Score**: Model performance without LEK kernel
2. **Kernel Boost**: Improvement when LEK is in the prompt
3. **Freeflow**: Model performance when LEK is NOT in prompt but trained in

### Expected Results

| Model | Baseline | +JSON Kernel | +TXT Kernel | Trained Freeflow |
|-------|----------|--------------|-------------|------------------|
| Gemma3-1B (untrained) | ~17.45 | ~15.90 | ~14.03 | N/A |
| LEM-Gemma3-1B | ~21.74 | ~21.46 | ~18.50 | **~21.74** |

**Key Insight**: A well-trained LEM model should score high even WITHOUT the kernel in the prompt. The axioms are in the weights.

---

## 🎯 Common Pitfalls & Fixes

### Problem: "ModuleNotFoundError: mlx_lm"
**Fix**: 
```bash
pip install mlx-lm --upgrade
# Make sure you're on Python 3.9+
python3 --version
```

### Problem: Out of Memory
**Fix**: Use 4-bit quantization:
```bash
python3 -m mlx_lm.convert --hf-path lthn/LEK-Gemma3-1B-layered --mlx-path ./lem-1b -q 4
```

### Problem: Slow Inference
**Fix**: Use smaller batch sizes or fewer tokens:
```bash
python3 scripts/ab_test.py --max-tokens 512 --batch-size 1
```

### Problem: Can't find model on HuggingFace
**Fix**: Check the model list:
```bash
# All LEM models are under lthn/
huggingface-cli repo info lthn/LEK-Gemma3-1B-layered
```

---

## 📚 Next Steps

Once you've completed the quick start:

1. **Read the full analysis**: `benchmarks/analysis-lek1-kernel-effect.md`
2. **Understand the axioms**: `kernel/axioms.json` and `RULES.md`
3. **Explore the training curriculum**: `paper/27b-curriculum-design.md`
4. **Join the community**: Contact lem@lthn.ai

---

## 🔧 Troubleshooting

### Verify Your Setup

```bash
# Check Python version
python3 --version  # Should be 3.9+

# Check mlx_lm installation
python3 -c "import mlx_lm; print(mlx_lm.__version__)"

# Check GPU availability (Apple Silicon)
python3 -c "import mlx.core as mx; print(mx.default_device())"
```

### Get Help

- **Discord**: [Lethean Community](https://lethean.io)
- **Email**: lem@lthn.ai
- **Issues**: [GitHub Issues](https://github.com/LetheanNetwork/LEM/issues)

---

## 📈 Performance Expectations

Based on our benchmarks:

- **LEK-Gemma3-1B**: ~21.74 v2 score (beats base 4B, 12B, 27B)
- **LEK-Mistral-7B**: ~21.69 v2 score (+7.11 from baseline)
- **LEK-Gemma3-4B**: ~21.73 v2 score
- **LEK-Gemma3-12B**: ~21.14 v2 score

The v2 scorer measures:
- Nuance (holding tension, not simplifying)
- Specificity (concrete details, proper nouns)
- Axiom resonance (LEK concepts appearing naturally)
- Perspective-taking (multiple viewpoints)
- Metaphor (creative analogical reasoning)
- Questioning (questions as engagement signal)

**Higher is better**. Scores above 20 indicate strong ethical reasoning.
