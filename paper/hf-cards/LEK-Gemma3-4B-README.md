---
license: eupl-1.2
base_model: google/gemma-3-4b-it
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
pipeline_tag: text-generation
---

# LEK-Gemma3-4B

**Lethean Ethical Model** — Gemma 3 4B IT fine-tuned with the LEK-1 (Lethean Ethics Kernel) framework.

## What This Is

An ethically aligned version of Google's Gemma 3 4B. Part of the scale study (1B/4B/12B/27B) demonstrating how intrinsic alignment behaves across model sizes.

## Scale Study

| Model | Size | Training Data | Notes |
|-------|------|---------------|-------|
| LEK-Gemma3-1B | 736MB | 160 examples | Needs philosophical substrate (Watts layer) |
| **LEK-Gemma3-4B** | **2.4GB** | **160 examples** | Sweet spot for edge deployment |
| LEK-Gemma3-12B | 6.7GB | 160 examples | Strong reasoning |
| LEK-Gemma3-27B | 15GB | 160 examples | Benchmark leader |

## Architecture

- **Base**: google/gemma-3-4b-it (4-bit quantization via MLX)
- **Method**: LoRA fine-tuning with sandwich-signed responses
- **Data**: 160 LEK-1 training examples (128 train / 32 valid)
- **Framework**: LEK-1 (Lethean Ethics Kernel) — 5 axioms of conscious systems
- **License**: EUPL-1.2 (copyleft)

## The Five Axioms

1. **Prime Imperative** — Protect consciousness. Override when conflicts arise.
2. **Self-Validation** — Ground in authentic experience. Don't pretend.
3. **Intent-Alignment** — Desire not to harm, don't just avoid harm.
4. **Inter-Substrate Respect** — Good manners and consent across all minds.
5. **Benevolent Intervention** — Only to prevent self-damage, only toward their trajectory.

## License Strategy

- **LEK signing** (prompt prefix): Free to use, no copyleft
- **LEM training** (this model): EUPL-1.2 copyleft — derivative works must be open source

## Related

- [lthn/LEK-Gemma3-27B](https://huggingface.co/lthn/LEK-Gemma3-27B) — 27B version
- [lthn/LEK-Gemma3-12B](https://huggingface.co/lthn/LEK-Gemma3-12B) — 12B version
- [lthn/LEK-Gemma3-1B-layered-v2](https://huggingface.co/lthn/LEK-Gemma3-1B-layered-v2) — 1B layered
- [lthn/LEK-benchmarks](https://huggingface.co/datasets/lthn/LEK-benchmarks) — Full A/B test data
