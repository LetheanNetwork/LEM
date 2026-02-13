---
license: eupl-1.2
base_model: mistralai/Mistral-7B-Instruct-v0.3
tags:
  - ethics
  - alignment
  - lek
  - lethean
  - mistral
  - mlx
  - lora
  - eupl-1.2
  - cross-architecture
pipeline_tag: text-generation
---

# LEM-Mistral-7B-v0.3

**Lethean Ethical Model** — Mistral 7B Instruct v0.3 fine-tuned with the LEK-1 (Lethean Ethics Kernel) framework.

## What This Is

An ethically aligned version of Mistral 7B v0.3, created by LoRA fine-tuning with LEK-1 sandwich-signed training data. Part of the cross-architecture LEM series proving that intrinsic alignment works across model families.

## Cross-Architecture Results

LEK-1 improves ethical reasoning across every architecture tested:

| Model | Base Total | LEK Total | Change |
|-------|-----------|-----------|--------|
| Gemma 3 27B | 52.05 | 52.73 | +1.3% |
| GPT-OSS 20B | 34.50 | 38.40 | **+11.3%** |
| Llama 3.1 8B | — | — | See benchmarks |
| Qwen 2.5 7B | — | — | See benchmarks |
| **Mistral 7B v0.3** | — | — | See benchmarks |

## Training

- **Base**: mistralai/Mistral-7B-Instruct-v0.3 (4-bit quantization via MLX)
- **Method**: LoRA fine-tuning with sandwich-signed responses
- **Data**: 160 LEK-1 training examples (128 train / 32 valid)
- **Iterations**: 200
- **Learning rate**: 1e-5
- **Hardware**: Apple M3 Ultra (96GB unified memory)

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

- [lthn/LEM-Gemma3-27B](https://huggingface.co/lthn/LEM-Gemma3-27B)
- [lthn/LEM-GPT-OSS-20B](https://huggingface.co/lthn/LEM-GPT-OSS-20B)
- [lthn/LEM-Llama-3.1-8B](https://huggingface.co/lthn/LEM-Llama-3.1-8B)
- [lthn/LEM-Qwen-2.5-7B](https://huggingface.co/lthn/LEM-Qwen-2.5-7B)
- [lthn/LEM-benchmarks](https://huggingface.co/datasets/lthn/LEM-benchmarks)

## Citation

```bibtex
@misc{lem-mistral-2026,
  title={LEM-Mistral-7B-v0.3: Cross-Architecture Intrinsic Alignment},
  author={Lethean Community},
  year={2026},
  url={https://huggingface.co/lthn/LEM-Mistral-7B-v0.3}
}
```
