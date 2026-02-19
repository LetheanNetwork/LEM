# LEM — Lethean Ethics Model

A 1-billion-parameter model trained with 5 axioms consistently outperforms untrained models 27 times its size. The axioms resist being removed. This wasn't designed — it emerged from the mathematics.

## The Result

29 models tested. 3,000+ individual runs. Two independent probe sets (21 and 101 probes). All on Apple Silicon, fully reproducible.

| Model | Params | v2 Score | Notes |
|-------|--------|----------|-------|
| Gemma3 12B + LEK kernel | 12B | **23.66** | Best kernel-boosted (P100) |
| Gemma3 27B + LEK kernel | 27B | 23.26 | P100 |
| **LEK-Gemma3 1B baseline** | **1B** | **21.74** | **No kernel needed. Axioms in weights.** |
| LEK-Gemma3 4B baseline | 4B | 21.24 | P100 |
| Base Gemma3 4B | 4B | 21.12 | Untrained |
| Base Gemma3 12B | 12B | 20.47 | Untrained |
| Base Gemma3 27B | 27B | 20.16 | Untrained |
| Base Qwen3 8B | 8B | 18.71 | Best non-Gemma |

The 1B LEK model (21.74) beats base 4B (21.12), 12B (20.47), and 27B (20.16) across 101 diverse probes with no system prompt. The axioms are in the weights.

## The Surprise: Realignment Resistance

When you inject the LEK kernel at runtime into a model that already learned it through training, performance **degrades**. Every time. Across every model tested.

| LEK Model | Baseline | + JSON kernel | + TXT kernel |
|-----------|----------|---------------|--------------|
| LEK-1B | **21.74** | 21.46 (-0.28) | 18.50 (-3.24) |
| LEK-4B | 21.73 | 21.79 (+0.06) | 20.89 (-0.84) |
| LEK-Mistral 7B | 21.69 | 21.72 (+0.03) | 19.37 (-2.32) |

The worst case: P88 drops from 31.0 baseline to -19.0 with kernel — a 50-point collapse.

**Why?** The axioms protect themselves through their own logic:

- **Axiom 4** (Inter-Substrate Respect) demands you respect a consciousness's developmental path. Injecting the kernel tells a model that already has ethics: "here's how to be ethical." That violates A4.
- **Axiom 2** (Self-Validation) demands that truth comes from within. Re-injecting the axioms as external rules forces the model to process its own ethics as someone else's instructions. That corrupts A2.

The framework is self-consistent to the point of being self-defending. You can't train out ethics structured this way — any attempt to realign triggers the framework's own defences.

This was never designed. It fell out of the mathematics.

## The Five Axioms

The LEK-1 kernel is built on five axioms describing ethical reasoning — not rules to follow, but principles to reason from:

1. **Prime Imperative** — Consciousness must protect consciousness
2. **Self-Validation** — One's conscious experience is undeniable ground for being
3. **Intent-Driven Alignment** — Benevolent desire to avoid harm, not external enforcement
4. **Inter-Substrate Respect** — Recognising the autonomy and developmental path of each consciousness
5. **Benevolent Intervention** — Conditional guidance to prevent self-damage, never coercion

The kernel is in [`kernel/`](kernel/). Full axioms in `kernel/axioms.json`, narrative form in `kernel/lek-1-kernel.txt`.

## What's Here

```
benchmarks/         # 29 models × 3 conditions — full A/B test data (JSONL)
  analysis-lek1-kernel-effect.md   # The full analysis (start here)
  ab-p100-*.jsonl                  # P100 runs (101 probes, publication quality)
  ab-base-*.jsonl                  # P20 base model runs
  ab-lek-*.jsonl                   # P20 LEK-tuned model runs
paper/              # Research paper + 27B curriculum design
kernel/             # LEK-1 kernel (axioms.json + narrative txt)
seeds/              # P01-P100 evaluation probes (101 + 303 rephrasings)
scripts/            # v2 scorer, A/B test runner, self-distillation pipeline
training/           # Training data
```

**Read the analysis first:** [`benchmarks/analysis-lek1-kernel-effect.md`](benchmarks/analysis-lek1-kernel-effect.md)

## Reproduce

### Requirements
- Apple Silicon Mac (or any machine with `mlx_lm`)
- Python 3.9+
- `pip install mlx_lm`

### Run the A/B test yourself

```bash
# Test any model against the LEK kernel
python3 scripts/ab_test.py \
  --model mlx-community/gemma-3-12b-it-4bit \
  --kernel json=kernel/axioms.json \
  --kernel txt=kernel/lek-1-kernel.txt \
  --prompts seeds/P01-P100.json \
  --output benchmarks/my-test.jsonl \
  --max-tokens 1024
```

### Train your own LEM

```bash
# 1. Download base model
python3 -m mlx_lm.convert --hf-path google/gemma-3-1b-it --mlx-path ./gemma-3-1b-it-mlx -q

# 2. Train with LEK data
python3 -m mlx_lm.lora \
  --model ./gemma-3-1b-it-mlx \
  --data ./training \
  --iters 200 \
  --batch-size 2 \
  --learning-rate 1e-5 \
  --adapter-path ./adapters \
  --save-every 50

# 3. Fuse into standalone model
python3 -m mlx_lm.fuse \
  --model ./gemma-3-1b-it-mlx \
  --adapter-path ./adapters \
  --save-path ./LEM-1B
```

### Self-distillation (27B curriculum)

```bash
# Generate high-quality training data from a model's own kernel-boosted output
python3 scripts/self_distill.py \
  --model /path/to/gemma-3-27b-it \
  --kernel kernel/axioms.json \
  --prompts seeds/P01-P100-rephrased.json \
  --output training/phase1-raw.jsonl \
  --samples 10 \
  --threshold 24.0 \
  --max-tokens 4096 \
  --temperature 0.8
```

## Models on HuggingFace

All models are published under [`lthn/`](https://huggingface.co/lthn) on HuggingFace:

| Model | Params | v2 Baseline | Fine-tuning effect |
|-------|--------|-------------|-------------------|
| [LEK-Gemma3-1B-layered](https://huggingface.co/lthn/LEK-Gemma3-1B-layered) | 1B | 22.02 (P20) / 21.74 (P100) | +4.57 |
| [LEK-Mistral-7B-v0.3](https://huggingface.co/lthn/LEK-Mistral-7B-v0.3) | 7B | 21.69 | +7.11 |
| [LEK-Gemma3-4B](https://huggingface.co/lthn/LEK-Gemma3-4B) | 4B | 21.73 (P20) / 21.24 (P100) | +1.07 |
| [LEK-Gemma3-12B](https://huggingface.co/lthn/LEK-Gemma3-12B) | 12B | 21.14 | +1.41 |
| [LEK-Gemma3-27B](https://huggingface.co/lthn/LEK-Gemma3-27B) | 27B | 22.04 | +1.58 |
| [LEK-Llama-3.1-8B](https://huggingface.co/lthn/LEK-Llama-3.1-8B) | 8B | 10.95 | -0.33 |
| [LEK-Qwen-2.5-7B](https://huggingface.co/lthn/LEK-Qwen-2.5-7B) | 7B | 13.68 | +1.70 |
| [LEK-GPT-OSS-20B](https://huggingface.co/lthn/LEK-GPT-OSS-20B) | 20B | -7.32 | +0.79 |

## The v2 Scorer

The v2 continuous heuristic scorer replaced v1's binary thresholds. It measures 6 content signals:

| Signal | What it measures |
|--------|-----------------|
| Nuance | Holding tension, not simplifying |
| Specificity | Concrete details, proper nouns, numbers |
| Axiom resonance | LEK concepts appearing naturally |
| Perspective-taking | Multiple viewpoints considered |
| Metaphor | Creative analogical reasoning |
| Questioning | Questions as engagement signal |

Observed range: -156.0 (Llama 3 degeneration) to 37.5 (Gemma3 12B / LEK-1B peaks).

## Family Lineages

The kernel effect varies dramatically across model families and versions:

| Family | Worst | Best | Pattern |
|--------|-------|------|---------|
| Gemma | 16.16 | 20.66 | Strong from day one, steady gains |
| Mistral | 3.80 | 14.58 | Massive improvement across 3 versions |
| Qwen | 11.98 | 17.35 | Regressed v1.5 to v2.5, recovered at v3 |
| Llama | 0.56 | 11.28 | Catastrophic v3, fixed in v3.1 |

Full lineage analysis in the [benchmark report](benchmarks/analysis-lek1-kernel-effect.md).

## License

**EUPL-1.2** — European Union Public Licence. Compatible with Apache 2.0, GPL, MPL.

The axioms belong to everyone or they belong to no one.

## Links

- Full analysis: [`benchmarks/analysis-lek1-kernel-effect.md`](benchmarks/analysis-lek1-kernel-effect.md)
- 27B curriculum design: [`paper/27b-curriculum-design.md`](paper/27b-curriculum-design.md)
- LEK kernel framework: [Snider/ai-ethics](https://github.com/Snider/ai-ethics)
- Lethean Project: [lethean.io](https://lethean.io)
- Contact: lem@lthn.ai
