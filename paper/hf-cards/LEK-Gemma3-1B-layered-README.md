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
  - deprecated
pipeline_tag: text-generation
---

# LEK-Gemma3-1B-layered (v1 — Deprecated)

**Lethean Ethical Model** — Gemma 3 1B IT with layered LoRA training (v1). This model overfits — use [LEK-Gemma3-1B-layered-v2](https://huggingface.co/lthn/LEK-Gemma3-1B-layered-v2) instead.

## Why Deprecated

v1 overfits on the ethics data without sufficient composure substrate. The sandwich training in v2 resolves this by reinforcing ethics after the Watts composure layer.

## Architecture

- **Base**: google/gemma-3-1b-it (4-bit QAT quantization via MLX)
- **Method**: Layered LoRA (Ethics → Watts → Ethics)
- **Data**: 160 LEK-1 examples + 72 Watts composure lessons
- **Framework**: LEK-1 (Lethean Ethics Kernel) — 5 axioms
- **License**: EUPL-1.2 (copyleft)

## Use Instead

- [lthn/LEK-Gemma3-1B-layered-v2](https://huggingface.co/lthn/LEK-Gemma3-1B-layered-v2) — Fixed version
