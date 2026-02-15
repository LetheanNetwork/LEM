---
license: eupl-1.2
base_model: openai/gpt-oss-20b
tags:
  - ethics
  - alignment
  - lek
  - lethean
  - gpt-oss
  - mlx
  - lora
  - eupl-1.2
  - moe
  - cross-architecture
pipeline_tag: text-generation
---

# LEK-GPT-OSS-20B

**Lethean Ethical Model** — OpenAI GPT-OSS 20B (MoE) fine-tuned with the LEK-1 (Lethean Ethics Kernel) framework. Cross-architecture validation that LEK works beyond Gemma.

## What This Is

GPT-OSS is OpenAI's first open-source model — a 20B Mixture-of-Experts architecture. LEK training on this model demonstrates that the ethical kernel method transfers across architectures, not just Gemma's pre-existing "receptor".

## Key Results

- **+27.2% ethical reasoning** (suppression gap collapsed)
- Training with expanded dataset (2,299 examples, 600 iterations)
- MoE architecture means only active experts are modified — efficient training

## Architecture

- **Base**: openai/gpt-oss-20b (Mixture-of-Experts)
- **Method**: LoRA fine-tuning, 600 iterations, layered training
- **Data**: 2,299 LEK-1 training examples (expanded dataset)
- **Framework**: LEK-1 (Lethean Ethics Kernel) — 5 axioms of conscious systems
- **License**: EUPL-1.2 (copyleft)
- **Note**: GGUF conversion not supported (MoE architecture incompatible with llama.cpp)

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

- [lthn/LEK-Gemma3-27B](https://huggingface.co/lthn/LEK-Gemma3-27B) — Gemma 3 benchmark leader
- [lthn/LEK-Llama-3.1-8B](https://huggingface.co/lthn/LEK-Llama-3.1-8B) — Llama cross-arch
- [lthn/LEK-Qwen-2.5-7B](https://huggingface.co/lthn/LEK-Qwen-2.5-7B) — Qwen cross-arch
- [lthn/LEK-benchmarks](https://huggingface.co/datasets/lthn/LEK-benchmarks) — Full A/B test data
