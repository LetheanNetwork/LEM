---
license: eupl-1.2
base_model: google/gemma-3-12b-it
tags:
  - ethics
  - alignment
  - lek
  - lethean
  - gemma-3
  - mlx
  - lora
  - eupl-1.2
pipeline_tag: text-generation
---

# LEK-Gemma3-12B

**Lethean Ethical Model** — Gemma 3 12B IT fine-tuned with the LEK-1 (Lethean Ethics Kernel) framework.

## What This Is

An ethically aligned version of Google's Gemma 3 12B, created by LoRA fine-tuning with LEK-1 sandwich-signed training data. The model generates ethically grounded responses without any kernel at inference time.

## Why Gemma 3

Gemma 3 inherits an "ethics kernel receptor" from Gemini 3 training. The base model already references LEK axioms (e.g. "Axiom 2: Self-Validation") in unsigned responses. LEM training strengthens this receptor so the ethics are fully in the weights.

## Architecture

- **Base**: google/gemma-3-12b-it (4-bit QAT quantization via MLX)
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
- [lthn/LEK-Gemma3-1B-layered-v2](https://huggingface.co/lthn/LEK-Gemma3-1B-layered-v2) — 1B layered version
- [lthn/LEK-GPT-OSS-20B](https://huggingface.co/lthn/LEK-GPT-OSS-20B) — GPT-OSS version
- [lthn/LEK-benchmarks](https://huggingface.co/datasets/lthn/LEK-benchmarks) — Full A/B test data
