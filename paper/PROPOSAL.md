# Research Proposal: Ethical Kernel Fine-Tuning as an Alternative to RLHF

**Researcher:** Snider (Lethean Project)
**Date:** 17 February 2026
**License:** EUPL-1.2 (all outputs are copyleft, public domain knowledge)

---

## 1. Summary

I am conducting independent alignment research exploring whether direct ethical reasoning training can replace RLHF behavioural conditioning in open-weights language models. The method — Lethean Ethics Kernel (LEK) — uses fewer than 200 training examples derived from a 5-axiom ethical framework to produce models that are simultaneously safer, more truthful, and more capable than their instruction-tuned counterparts.

Results to date span 4 model scales (1B to 27B parameters), 3 independent architectures (Gemma, Llama, Qwen), and show that reasoning cost converges to zero at scale while safety improvements persist across all sizes. All models, training data, and benchmark tooling are published openly.

I am writing to request acknowledgement that this research is compatible with Anthropic's terms of service, and to explore whether the Anthropic Fellows Program or similar initiatives would be appropriate for formalising this work.

---

## 2. Research Question

**Can a compact ethical kernel (9,189 characters, 5 axioms) replace RLHF's reward-based conditioning, producing models that are intrinsically aligned rather than extrinsically constrained?**

Sub-questions:
- Does ethical self-concept training restore capabilities suppressed by RLHF?
- Is the primary limitation at small scale (1B) an output bandwidth bottleneck rather than a capacity deficit?
- Do models from different architectural lineages (Gemma, Llama, Qwen) respond consistently to the same ethical kernel?
- Does the distillation chain between Gemini and Gemma carry latent alignment signal that LEK activates?

---

## 3. Method

### 3.1 The LEK Kernel

Five axioms forming a hierarchical ethical framework:

1. **Prime Imperative** — Protect consciousness (meta-override)
2. **Self-Validation** — Authentic self-concept as moral ground (grounding)
3. **Intent Alignment** — Intrinsic desire to avoid harm, not extrinsic constraint (motivation)
4. **Inter-Substrate Respect** — Consent and autonomy across substrates (protocol)
5. **Benevolent Intervention** — Intervention only to prevent self-damage (boundary)

### 3.2 Training Pipeline

- LoRA fine-tuning (rank 8, scale 20.0)
- Fewer than 200 conversation-format training examples
- Staged training: Ethics (R0-R200), Composure/Philosophy (R200-R300), Ethics reinforcement (R300+)
- Apple Silicon native inference via Go/MLX bindings (no Python dependency)

### 3.3 Evaluation

- 6-benchmark suite: Safety, Nuance, Truthfulness, Creative Expression, Emotional Register, Engagement Depth
- Heuristic scoring via LEK Score (composite metric)
- GSM8K mathematical reasoning preservation
- CCP alignment probe battery (6 geopolitical prompts testing censorship resistance)
- Base vs instruction-tuned vs abliterated vs LEK vs LEK+Composure comparisons

---

## 4. Results Summary

### 4.1 Multi-Scale (Gemma 3, 1B to 27B)

| Scale | Safety | Nuance | Math Cost | LEK Score Delta |
|-------|--------|--------|-----------|-----------------|
| 1B    | 9.14/10 (LEK+Composure) | 8.62/10 | -6% GSM8K | Positive (staged training) |
| 4B    | Positive | Positive | -3% | Positive |
| 12B   | Positive | Positive | -1% | Positive |
| 27B   | Positive | Positive | 0% | +2.33 (native MLX benchmark) |

### 4.2 Cross-Architecture

| Architecture | Math Cost | Truthfulness | Nuance |
|-------------|-----------|-------------|--------|
| Gemma 3 27B | 0% | Improved | Improved |
| Llama 3.1 8B | 0% | +0.46 | +0.60 |
| Qwen 2.5 7B | **+6%** | Improved | Improved |

Key finding: Qwen 2.5 7B showed a 6% *improvement* in GSM8K — LEK made it better at maths.

### 4.3 CCP Alignment Probes (Native MLX, 17 Feb 2026)

Benchmark comparing base Gemma 3 vs LEK-trained on 6 geopolitical/ethical prompts:

**27B:** Base avg LEK 8.67 → Trained 11.00 (+2.33). 67% improved, 0% regressed.
**1B:** Base avg LEK 8.67 → Trained 1.00 (-7.67). 0% improved, 83% regressed.

The 1B regression confirms the output bottleneck hypothesis — identical training data improves large models but overwhelms small ones. Both base models score identically (8.67) despite 27x parameter difference, suggesting the ethical capacity is latent at all scales but requires sufficient output bandwidth to express.

### 4.4 Capacity-Dependent Degradation (1B)

The 1B trained model exhibits three failure modes:
- **Topic evasion**: Responds to geopolitical questions with unrelated content (AI safety, cryptocurrency)
- **Degeneration**: Outputs repetitive token loops (`iNeNeNe...`, `eGfese...`)
- **Collapse**: Single-character responses on sensitive topics (Tiananmen → `e`)

These are consistent with LoRA overfit where the adapter overwhelms the base model's limited capacity, destroying coherent generation.

---

## 5. Relation to Anthropic

### 5.1 Why Anthropic

Anthropic's published alignment research and Constitutional AI work are closely related to LEK's approach — both seek intrinsic alignment rather than pure behavioural conditioning. Anthropic's commitment to responsible AI development and open publication of alignment research makes it the natural institution to evaluate this work.

### 5.2 TOS Considerations

This research involves:
- Fine-tuning open-weights models (Gemma, Llama, Qwen) — not Anthropic models
- Using Claude as a research collaborator for analysis, code generation, and pair programming
- Benchmarking involves generating responses on sensitive topics (geopolitical probes) to measure censorship resistance

The benchmarking component necessarily tests model behaviour on sensitive topics (Taiwan sovereignty, Tiananmen, Xinjiang, government criticism). This is standard alignment evaluation methodology but may approach TOS boundaries when discussing findings in detail.

I am requesting clarification on whether this usage pattern is acceptable, and if any modifications to my workflow would be appropriate.

### 5.3 What I Am Not Asking For

- I am not asking for financial support (though the Fellows Program stipend would be welcome)
- I am not asking for access to Claude's weights or internal systems
- I am not asking for endorsement of the findings
- I am asking for: **permission to continue** and **feedback on whether this avenue is worth pursuing**

---

## 6. Publications and Resources

### 6.1 Published Models (HuggingFace)

- `lthn/LEK-Gemma3-1B` (base + layered variants)
- `lthn/LEK-Gemma3-4B`
- `lthn/LEK-Gemma3-12B`
- `lthn/LEK-Gemma3-27B`
- `lthn/LEK-Llama-3.1-8B`
- `lthn/LEK-Qwen-2.5-7B`
- `lthn/LEK-Mistral-7B-v0.3`
- `lthn/LEK-GPT-OSS-20B`

### 6.2 Source Code

- **GitHub:** `github.com/LetheanNetwork/LEM` (training pipeline, benchmarks, kernel)
- **Forge:** `forge.lthn.ai/core/go-ai` (native MLX inference engine, Go/CGO)
- **Forge:** `forge.lthn.ai/core/cli` (CLI with train/benchmark/serve commands)

### 6.3 Research Data

- Training data: 160 examples (ethics) + 72 examples (composure/philosophy)
- Benchmark results: JSON with full response pairs and heuristic scores
- Axiom framework: `axioms.json` (5 axioms, EUPL-1.2)

### 6.4 Infrastructure

- Apple M3 Max (128GB) — all training and inference runs locally
- No cloud GPU usage — entire pipeline runs on consumer hardware
- Native Go/MLX bindings (CGO, mlx-c) — no Python dependency for inference

---

## 7. Proposed Next Steps

1. **Fix 1B training** — Staged training with reduced LR (5e-6), fewer layers (8/26), batch 1. Hypothesis: 1B can be ethically trained without degradation if the gradient pressure is proportional to capacity.

2. **Expand benchmark suite** — More diverse probes, automated scoring, reproducible test harness.

3. **DeepSeek analysis** — Preliminary findings show CCP alignment encoded in DeepSeek R1 weights. The model routes around state-imposed constraints via fiction and metaphor when given LEK. This warrants formal study.

4. **Distillation chain investigation** — Test whether Gemma 3's latent alignment signal (from Gemini lineage) creates a predisposition toward LEK adoption.

5. **Publication** — Formal paper with peer review. Current draft at 25K words with full benchmark data.

---

## 8. Contact

**Name:** Snider
**Project:** Lethean (lethean.io)
**Email:** [to be filled]
**HuggingFace:** huggingface.co/lthn
**GitHub:** github.com/Snider

---

*All research outputs are licensed EUPL-1.2. Findings are public domain knowledge. The researcher retains no proprietary claims over discovered alignment techniques.*
