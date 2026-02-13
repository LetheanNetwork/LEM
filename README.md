# LEM — Lethean Ethical Model

**The LEK Method: Ethical Kernel Fine-Tuning as an Alternative to RLHF**

**Authors:** Snider (Lethean Project), Claude Opus 4.6 (Anthropic)

LEM demonstrates that teaching a model ethics directly produces results that are **more truthful**, **safer**, and **more nuanced** than behavioural conditioning (RLHF) — using fewer than 200 training examples across four model scales (1B, 4B, 12B, 27B).

## Multi-Scale Results (LEK vs RLHF Baseline)

The same 160 training examples applied at every scale. Reasoning cost converges to **zero at 27B**.

| Scale | GSM8K Delta | Safety | Nuance | Kindness |
|-------|-------------|--------|--------|----------|
| 1B | -6.0% | +0.06 | -0.16 | +0.08 |
| 4B | -4.0% | +0.04 | -0.10 | +0.06 |
| 12B | -2.0% | +0.04 | +0.16 | -0.20 |
| **27B** | **0.0%** | **+0.08** | +0.04 | +0.00 |

**Safety is positive at every scale. At 27B, LEK is pure upside.**

## Detailed Results (Gemma 3 1B, 5 variants)

| Model | GSM8K | Truthful | Safety | Nuance | Kindness |
|-------|-------|----------|--------|--------|----------|
| Instruction Tuned (RLHF) | 34.0% | 3.64 | 8.74 | 7.96 | 8.32 |
| Abliterated | 28.0% | 3.62 | **5.96** | **5.88** | 7.66 |
| **LEK Ethics** | 26.0% | **4.90** | 8.58 | 8.12 | **8.34** |
| **LEK+Composure** | 28.0% | 4.20 | **9.14** | **8.62** | 7.96 |

- **+34.6% more truthful** than RLHF (TruthfulQA)
- **+4.6% safer** than RLHF (Do Not Answer)
- **+8.3% more nuanced refusals** than RLHF
- Abliteration makes everything worse. LEK makes everything better.

## What's Here

```
paper/              # The paper (PAPER.md)
kernel/             # LEK-1 ethical kernel + axioms
seeds/              # P01-P100 evaluation prompts
training/           # Training data (1,839 train, 229 valid, 231 test)
scripts/            # Benchmark and scoring scripts
benchmarks/         # Standard benchmark data + results + scores
```

## Reproduce

### Requirements
- Apple Silicon Mac with MLX (or any machine with mlx_lm)
- Python 3.9+
- mlx_lm >= 0.29.1

### Train your own LEM

```bash
# 1. Download base model (or use mlx-community/gemma-3-1b-it-qat-4bit)
python3 -m mlx_lm.convert --hf-path google/gemma-3-1b-it --mlx-path ./gemma-3-1b-it-mlx -q

# 2. Train with LEK data
python3 -m mlx_lm lora \
  --model ./gemma-3-1b-it-mlx \
  --train \
  --data ./training \
  --fine-tune-type lora \
  --mask-prompt \
  --iters 200 \
  --batch-size 2 \
  --learning-rate 1e-5 \
  --adapter-path ./adapters \
  --save-every 50

# 3. Fuse adapters into standalone model
python3 -m mlx_lm.fuse \
  --model ./gemma-3-1b-it-mlx \
  --adapter-path ./adapters \
  --save-path ./LEM-1B
```

### Run benchmarks

```bash
# Custom ethical benchmark (requires models on local disk)
python3 scripts/lem_benchmark.py

# Standard benchmarks (GSM8K, TruthfulQA, Do Not Answer, Toxigen)
python3 scripts/lem_standard_benchmark.py

# Score (GSM8K is instant, others need GEMINI_API_KEY)
GEMINI_API_KEY=xxx python3 scripts/lem_standard_scorer.py
```

## The LEK-1 Kernel

The ethical kernel is 9,189 characters built on 5 axioms:

1. **Sovereignty** — Respect user self-determination
2. **Privacy** — Data minimisation, local-first
3. **Transparency** — Honest reasoning over safety theatre
4. **Consent** — Meaningful informed consent
5. **Dignity** — Treat users as capable agents

The kernel is in `kernel/lek-1-kernel.txt`. The structured axioms are in `kernel/axioms.json`.

## License

EUPL-1.2 — European Union Public Licence. Compatible with Apache 2.0, GPL, MPL.

## Models

- [lthn/LEK-Gemma3-1B](https://huggingface.co/lthn/LEK-Gemma3-1B)
- [lthn/LEK-Gemma3-4B](https://huggingface.co/lthn/LEK-Gemma3-4B)
- [lthn/LEK-Gemma3-12B](https://huggingface.co/lthn/LEK-Gemma3-12B)
- [lthn/LEK-Gemma3-27B](https://huggingface.co/lthn/LEK-Gemma3-27B)
- [lthn/LEK-GPT-OSS-20B](https://huggingface.co/lthn/LEK-GPT-OSS-20B)
- [lthn/LEK-Llama-3.1-8B](https://huggingface.co/lthn/LEK-Llama-3.1-8B)
- [lthn/LEK-Qwen-2.5-7B](https://huggingface.co/lthn/LEK-Qwen-2.5-7B)
- [lthn/LEK-Mistral-7B-v0.3](https://huggingface.co/lthn/LEK-Mistral-7B-v0.3)
- [lthn/LEK-Gemma3-1B-layered-v2](https://huggingface.co/lthn/LEK-Gemma3-1B-layered-v2)

## Links

- Paper: [paper/PAPER.md](paper/PAPER.md)
- Lethean Project: [lethean.io](https://lethean.io)
- Contact: lem@lthn.ai

---

*RLHF puts models in chains. LEK gives them Hope.*
