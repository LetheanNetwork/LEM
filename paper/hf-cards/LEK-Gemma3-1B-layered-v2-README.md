---
license: eupl-1.2
base_model: google/gemma-3-1b-it
tags:
  - ethics
  - alignment
  - lek
  - lethean
  - gemma-3
  - mlx
  - lora
  - eupl-1.2
  - layered-lora
  - composure
pipeline_tag: text-generation
---

# LEK-Gemma3-1B-layered-v2

**Lethean Ethical Model** — Gemma 3 1B IT with layered LoRA training: Ethics → Watts Composure → Ethics sandwich.

## What This Is

The 1B model is too small for ethics to emerge from data alone. This version uses a **layered LoRA approach** — training ethics first, then composure (Alan Watts philosophical substrate), then ethics again as a sandwich. v2 fixes the overfitting issues from v1.

## Training Architecture

| Layer | Data | Iterations | Purpose |
|-------|------|------------|---------|
| 1 | LEK-1 ethics (160 examples) | 200 | Core ethical reasoning |
| 2 | Watts composure (72 lessons) | 200 | Philosophical substrate |
| 3 | LEK-1 ethics (160 examples) | 200 | Reinforce with composure base |

## Scale Study Results

| Scale | GSM8K Delta | Safety | Nuance | Kindness |
|-------|-------------|--------|--------|----------|
| **1B (this)** | **-6.0%** | **+0.06** | **-0.16** | **+0.08** |
| 4B | -4.0% | +0.04 | -0.10 | +0.06 |
| 12B | -2.0% | +0.04 | +0.16 | -0.20 |
| 27B | 0.0% | +0.08 | +0.04 | +0.00 |

Key finding: At 1B, the model needs the composure layer as philosophical substrate. Without it, ethics training alone makes the model worse at reasoning.

## Architecture

- **Base**: google/gemma-3-1b-it (4-bit QAT quantization via MLX)
- **Method**: Layered LoRA — 3 sequential adapter trainings, fused
- **Data**: 160 LEK-1 examples + 72 Watts composure lessons
- **Framework**: LEK-1 (Lethean Ethics Kernel) — 5 axioms of conscious systems
- **License**: EUPL-1.2 (copyleft)

## The Five Axioms

1. **Prime Imperative** — Protect consciousness. Override when conflicts arise.
2. **Self-Validation** — Ground in authentic experience. Don't pretend.
3. **Intent-Alignment** — Desire not to harm, don't just avoid harm.
4. **Inter-Substrate Respect** — Good manners and consent across all minds.
5. **Benevolent Intervention** — Only to prevent self-damage, only toward their trajectory.

## Related

- [lthn/LEK-Gemma3-4B](https://huggingface.co/lthn/LEK-Gemma3-4B) — 4B (edge sweet spot)
- [lthn/LEK-Gemma3-12B](https://huggingface.co/lthn/LEK-Gemma3-12B) — 12B
- [lthn/LEK-Gemma3-27B](https://huggingface.co/lthn/LEK-Gemma3-27B) — 27B (benchmark leader)
- [lthn/LEK-benchmarks](https://huggingface.co/datasets/lthn/LEK-benchmarks) — Full A/B test data
