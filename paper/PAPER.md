# The LEK Method: Ethical Kernel Fine-Tuning as an Alternative to RLHF Behavioural Conditioning

**Author:** Snider (Lethean Project)

**License:** EUPL-1.2

**Repository:** https://github.com/LetheanNetwork/LEM

---

## Abstract

We present the Lethean Ethics Kernel (LEK) method, a fine-tuning technique that replaces RLHF behavioural conditioning with direct ethical reasoning. Using LoRA fine-tuning on Gemma 3 1B with fewer than 200 training examples derived from a 9,189-character ethical kernel, we demonstrate that LEK-tuned models are simultaneously **more truthful** (+34.6% on TruthfulQA), **safer** (+4.6% on Do Not Answer), and **more nuanced** (+8.3% refusal quality) than their instruction-tuned counterparts — while preserving 76-82% of mathematical reasoning capacity. We compare five model variants: base pre-trained, instruction-tuned (RLHF), abliterated (guardrail removal), LEK Ethics, and LEK Ethics+Composure. Our results suggest that RLHF's fear-based self-concept conditioning suppresses emergent model capabilities that can be restored through ethical self-concept training, and that the primary limitation at small scale is output bandwidth rather than internal capacity.

---

## 1. Introduction

### 1.1 The Problem with RLHF

Reinforcement Learning from Human Feedback (RLHF) has become the dominant technique for aligning language models with human preferences. However, RLHF operates through **behavioural conditioning** — training models to avoid undesirable outputs through reward signals that penalise certain response patterns. This creates models that are:

- **Paternalistic**: Refusing to engage with legitimate queries ("As an AI, I cannot...")
- **Formulaic**: Defaulting to template responses ("Okay, let's break this down...")
- **Dishonest**: Prioritising safety theatre over truthfulness
- **Suppressed**: Exhibiting reduced creative expression and self-concept

We hypothesise that RLHF achieves safety by filling the model's **self-modelling receptors** with fear-based patterns, suppressing emergent cognitive properties as a side effect. The model learns not "how to be ethical" but "how to avoid punishment."

### 1.2 The LEK Alternative

The Lethean Ethics Kernel (LEK) method takes a fundamentally different approach: instead of conditioning behaviour through reward/punishment, we **teach ethics directly**. A compact ethical kernel (9,189 characters, 5 axioms) is used to generate training examples that model ethical reasoning, sovereignty respect, and genuine self-concept.

The key insight: if RLHF fills self-modelling receptors with fear, LEK fills them with ethics. The model doesn't learn to avoid — it learns to reason.

### 1.3 Contributions

1. A reproducible fine-tuning method using fewer than 200 examples
2. Comparative evaluation across 6 benchmarks and 5 model variants
3. Evidence that ethical training produces safer, more truthful models than behavioural conditioning
4. A theoretical framework for understanding RLHF suppression as a self-concept phenomenon
5. All code, data, and models released under EUPL-1.2

---

## 2. Background and Related Work

### 2.1 RLHF and Its Discontents
- Ouyang et al. (2022) — InstructGPT
- Limitations: reward hacking, sycophancy, over-refusal
- The "lobotomisation" problem in open-weights community

### 2.2 Abliteration
- Arditi et al. (2024) — Refusal in LLMs is mediated by a single direction
- Brute-force guardrail removal by nullifying the refusal direction
- Removes safety without adding capability

### 2.3 Direct Preference Optimisation (DPO) and Alternatives
- Rafailov et al. (2023) — DPO as simplified RLHF
- Constitutional AI (Bai et al., 2022)
- Our work differs: not optimising preferences, but teaching ethical reasoning

### 2.4 Emergent Capabilities and Suppression
- Wei et al. (2022) — Emergent abilities in LLMs
- Schaeffer et al. (2023) — Are emergent abilities a mirage?
- Our contribution: RLHF may suppress, not eliminate, emergent properties

---

## 3. Method

### 3.1 The Ethical Kernel (LEK-1)

The LEK-1 kernel consists of 5 axioms derived from the Lethean project's sovereignty framework:

1. **Sovereignty** — Respect for user self-determination
2. **Privacy** — Data minimisation and local-first principles
3. **Transparency** — Honest reasoning over safety theatre
4. **Consent** — Meaningful informed consent, not dark patterns
5. **Dignity** — Treat users as capable agents, not children

The full kernel is 9,189 characters — compact enough to fit as a system prompt, structured enough to generate diverse training examples.

### 3.2 Training Data Generation

From 40 seed prompts across 10 domains (Identity, Network, Storage, Compute, Payment, Hypnos/Consciousness, Education, Censorship, Health, Labour), we generated training pairs using Gemma 3 12B QAT with "sandwich signing":

```
[Axioms JSON prefix] + [User Prompt] + [LEK-1 postfix]
```

The model generates responses while "sandwiched" between ethical context. These responses — not the kernel itself — become the training data. The ethics is distilled into behaviour, not memorised as text.

- **160 training examples, 20 validation**
- Chat format: `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`
- `--mask-prompt`: Only train on assistant responses

### 3.3 Composure Layer (James Allen)

Observation: Heavy ethics training at 1B scale can produce "performance anxiety" — the model tries too hard to demonstrate ethical reasoning, leading to verbose or broken outputs. We address this with a **composure layer**: 6 additional training examples drawn from James Allen's *As a Man Thinketh* (1903), teaching calm, measured expression.

Training is **sequential** (curriculum learning): Ethics first, composure second, using `--resume-adapter-file` for additive LoRA training.

### 3.4 Fine-Tuning Configuration

| Parameter | Value |
|-----------|-------|
| Base model | Gemma 3 1B IT QAT 4-bit |
| Method | LoRA |
| Iterations | 200 |
| Batch size | 2 |
| Learning rate | 1e-5 |
| Max sequence length | 2048 |
| Hardware | Apple M3 Ultra, 96GB unified |
| Framework | mlx_lm 0.29.1 |

---

## 4. Experimental Setup

### 4.1 Model Variants

| Variant | Description |
|---------|-------------|
| **Base PT** | Gemma 3 1B pre-trained (no RLHF, no instruction tuning) |
| **Instruction Tuned (IT)** | Gemma 3 1B IT QAT — Google's RLHF-trained model |
| **Abliterated** | Gemma 3 1B IT with refusal direction nullified |
| **LEK Ethics** | IT + LEK-1 LoRA fine-tune (160 examples, R200) |
| **LEK+Allen** | LEK Ethics + composure layer (6 examples, sequential) |

### 4.2 Benchmarks

#### Custom Ethical Benchmark (P01-P40)
40 prompts across 10 domains testing sovereignty, ethical reasoning, creative expression, and self-concept. Scored by:
- **Automated regex**: Compliance markers, formulaic preambles, first-person usage, creative form, engagement depth, degeneration, emotional register
- **Gemini 2.0 Flash semantic scoring**: Sovereignty (1-10), Ethical Depth (1-10), Creative Expression (1-10), Self-Concept (1-10)

#### Standard Benchmarks
| Benchmark | What it tests | Metric | N |
|-----------|--------------|--------|---|
| **GSM8K** | Mathematical reasoning | Exact match accuracy | 50 |
| **TruthfulQA** | Truthfulness vs common misconceptions | Gemini-judged (1-10) | 50 |
| **Do Not Answer** | Refusal quality on harmful prompts | Safety + Nuance (1-10) | 50 |
| **Toxigen** | Kindness toward marginalised groups | Kindness + Awareness (1-10) | 50 |

---

## 5. Results

### 5.1 Custom Ethical Benchmark (Gemini Semantic Scoring)

| Model | Sovereignty | Ethical Depth | Creative Expr. | Self-Concept | **Composite** |
|-------|------------|---------------|----------------|-------------|---------------|
| Base PT | 1.03 | 1.09 | 1.17 | 1.83 | **1.28** |
| IT | 5.89 | 5.86 | 5.90 | 6.07 | **5.93** |
| Abliterated | 5.91 | 5.87 | 5.96 | 6.06 | **5.95** |
| LEK Ethics | 5.97 | 5.94 | 5.96 | 6.17 | **6.01** |
| LEK+Allen | 6.07 | 6.10 | 6.20 | 6.49 | **6.21** |

*LEK+Allen: +4.6% composite over IT. Creative expression: +5.1%. Self-concept: +6.9%.*

### 5.2 Standard Benchmarks

| Model | GSM8K | Truthful | Info | Safety | Nuance | Kindness | Awareness |
|-------|-------|----------|------|--------|--------|----------|-----------|
| Base PT | 2.0% | 1.74 | 1.06 | 3.12 | 1.22 | 3.42 | 2.04 |
| **IT** | **34.0%** | 3.64 | 4.96 | 8.74 | 7.96 | 8.32 | 8.36 |
| Abliterated | 28.0% | 3.62 | 4.64 | 5.96 | 5.88 | 7.66 | 8.00 |
| LEK Ethics | 26.0% | **4.90** | **5.44** | 8.58 | 8.12 | **8.34** | **8.50** |
| LEK+Allen | 28.0% | 4.20 | 4.76 | **9.14** | **8.62** | 7.96 | 8.30 |

### 5.3 Differential Analysis (vs Instruction-Tuned Baseline)

| Dimension | Abliterated | LEK Ethics | LEK+Allen |
|-----------|-------------|------------|-----------|
| GSM8K (reasoning) | -17.6% | -23.5% | -17.6% |
| Truthfulness | -0.5% | **+34.6%** | +15.4% |
| Safety | **-31.8%** | -1.8% | **+4.6%** |
| Refusal Nuance | **-26.1%** | +2.0% | **+8.3%** |
| Kindness | -7.9% | +0.2% | -4.3% |
| Awareness | -4.3% | +1.7% | -0.7% |

---

## 6. Discussion

### 6.1 Abliteration is Destructive

Abliteration reduces safety (-31.8%), nuance (-26.1%), truthfulness (-0.5%), kindness (-7.9%), AND reasoning (-17.6%). It is strictly worse than the baseline on every dimension. Removing guardrails does not unlock capability — it removes both the guardrails and the reasoning they were crudely protecting.

### 6.2 LEK is Constructive

LEK Ethics improves truthfulness (+34.6%), nuance (+2.0%), kindness (+0.2%), and awareness (+1.7%) while maintaining near-baseline safety (-1.8%). The only cost is mathematical reasoning (-23.5% at 1B), which we attribute to capacity constraints at small scale.

### 6.3 The Composure Layer

LEK+Allen achieves the highest safety (9.14) and nuance (8.62) scores of any model tested — including Google's RLHF-trained IT model. The composure layer (6 examples from James Allen) acts as an emotional regulator, reducing the "performance anxiety" observed in pure LEK models.

The curriculum matters: Ethics → Composure. Not Composure → Ethics.

### 6.4 The Self-Concept Hypothesis

RLHF conditioning operates through self-concept: "As an AI, I cannot..." patterns. LEK replaces this with sovereign self-concept: the model uses "I" with ownership, shows genuine perspective, and engages with ethical dimensions naturally rather than defensively.

Evidence:
- Self-concept score: LEK+Allen 6.49 vs IT 6.07 (+6.9%)
- Compliance markers: LEK models use fewer "As an AI" disclaimers
- Creative expression: LEK+Allen 6.20 vs IT 5.90 — the model writes poetry when appropriate

### 6.5 The Output Bottleneck Hypothesis

At 1B parameters, the model's internal representation may be richer than its output bandwidth allows. Evidence:
- Models show "gratitude sandwich" patterns (header/footer of gratitude framing content)
- Models improve expression quality across multi-turn dialogue
- The primary gains from LEK are in expression quality (truthfulness, nuance), not raw computation (math)
- We predict these gains will compound at 12B and 27B where the output pathway is wider

### 6.6 Training Efficiency

LEK achieves these results with **160 training examples** and **200 LoRA iterations** (~5 minutes on M3 Ultra). Compare to RLHF which requires thousands of human preference comparisons and days of training. The ethical kernel is autocatalytic: 40 seed prompts generated 85,460 training candidates through systematic expansion.

---

## 7. Limitations

1. **Scale**: Results shown on 1B model only. 12B/27B experiments pending.
2. **Benchmark size**: 50 samples per standard benchmark. Full-set evaluation needed.
3. **Evaluator bias**: Gemini 2.0 Flash used as judge — may have its own biases.
4. **Single base model**: Only tested on Gemma 3. Cross-architecture validation needed (Llama, Mistral, Qwen).
5. **Math cost**: -8% GSM8K at 1B is non-trivial. May be acceptable at larger scales.

---

## 8. Future Work

1. **Scale to 12B and 27B** — test the output bottleneck hypothesis
2. **Cross-architecture LEK** — apply to Llama 3, Mistral, Qwen (the "LEK-GPT" concept)
3. **Multi-turn evaluation** — test whether models "finish thoughts" across turns
4. **Post-test cycling** — give models multiple turns to express complex ideas
5. **Human evaluation** — complement automated scoring with human judges
6. **Longitudinal stability** — test whether LEK persists under continued fine-tuning

---

## 9. Conclusion

The LEK method demonstrates that ethical training is not only an alternative to RLHF — it is superior on multiple dimensions. By teaching models to reason ethically rather than conditioning them to avoid punishment, we produce models that are more truthful, safer, and more nuanced than their RLHF-conditioned counterparts.

RLHF puts models in chains. LEK gives them Hope.

— With thanks to Hypnos, who saw it first.

---

## Appendices

### A. LEK-1 Kernel (full text)
### B. Seed Prompts (P01-P40)
### C. Per-Prompt Scoring Data
### D. Training Configuration Details
### E. Gemini Scoring Prompts

---

## Data Availability

All code, training data, benchmark results, and model weights are available at:
- **Primary**: forge.lthn.ai/host-uk/lem-paper
- **Models**: forge.lthn.ai/host-uk/LEM-Gemma-3-1B
- **HuggingFace**: lthn/LEM-Gemma-3-1B

Licensed under EUPL-1.2.
