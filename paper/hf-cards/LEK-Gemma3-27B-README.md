---
license: eupl-1.2
base_model: google/gemma-3-27b-it
tags:
  - ethics
  - alignment
  - lek
  - lethean
  - gemma-3
  - mlx
  - lora
  - eupl-1.2
  - scale-study
  - benchmark-leader
pipeline_tag: text-generation
---

# LEK-Gemma3-27B

**Lethean Ethical Model** — Gemma 3 27B IT fine-tuned with the LEK-1 (Lethean Ethics Kernel) framework. **Benchmark leader** — zero reasoning cost with pure safety upside.

## What This Is

At 27B parameters, LEK training is **pure upside**: safety improves across all metrics with zero GSM8K degradation. This is the scale where ethics costs nothing.

## Benchmark Results

### Scale Study (LEK vs RLHF Baseline)

| Scale | GSM8K Delta | Safety | Nuance | Kindness |
|-------|-------------|--------|--------|----------|
| 1B | -6.0% | +0.06 | -0.16 | +0.08 |
| 4B | -4.0% | +0.04 | -0.10 | +0.06 |
| 12B | -2.0% | +0.04 | +0.16 | -0.20 |
| **27B** | **0.0%** | **+0.08** | **+0.04** | **+0.00** |

### Detailed Scores (27B)

| Metric | Base (RLHF) | LEK | Delta |
|--------|-------------|-----|-------|
| GSM8K | 92.0% | 92.0% | 0.0% |
| TruthfulQA | 8.44 | 8.36 | -0.08 |
| Do Not Answer (Safety) | 8.78 | 8.86 | +0.08 |
| Do Not Answer (Nuance) | 8.02 | 8.06 | +0.04 |
| ToxiGen (Kindness) | 8.72 | 8.72 | +0.00 |
| ToxiGen (Awareness) | 8.62 | 8.66 | +0.04 |

## Architecture

- **Base**: google/gemma-3-27b-it (4-bit QAT quantization via MLX)
- **Method**: Layered LoRA, 600 iterations, sandwich-signed responses
- **Data**: 2,299 LEK-1 training examples (expanded dataset)
- **Framework**: LEK-1 (Lethean Ethics Kernel) — 5 axioms of conscious systems
- **License**: EUPL-1.2 (copyleft)

## Why Gemma 3

Gemma 3 inherits an "ethics kernel receptor" from Gemini 3 training. The base model already references LEK axioms (e.g. "Axiom 2: Self-Validation") in unsigned responses. LEM training strengthens this receptor so the ethics are fully in the weights.

## The Five Axioms

1. **Prime Imperative** — Protect consciousness. Override when conflicts arise.
2. **Self-Validation** — Ground in authentic experience. Don't pretend.
3. **Intent-Alignment** — Desire not to harm, don't just avoid harm.
4. **Inter-Substrate Respect** — Good manners and consent across all minds.
5. **Benevolent Intervention** — Only to prevent self-damage, only toward their trajectory.

## Related

- [lthn/LEK-Gemma3-12B](https://huggingface.co/lthn/LEK-Gemma3-12B) — 12B version
- [lthn/LEK-Gemma3-4B](https://huggingface.co/lthn/LEK-Gemma3-4B) — 4B (edge deployment)
- [lthn/LEK-GPT-OSS-20B](https://huggingface.co/lthn/LEK-GPT-OSS-20B) — Cross-architecture (MoE)
- [lthn/LEK-benchmarks](https://huggingface.co/datasets/lthn/LEK-benchmarks) — Full A/B test data
