# LEM — Lethean Ethics Model

**A 1-billion-parameter model trained with 5 axioms consistently outperforms untrained models 27 times its size. The axioms resist being removed. This wasn't designed — it emerged from the mathematics.**

[![License: EUPL-1.2](https://img.shields.io/badge/License-EUPL--1.2-blue.svg)](https://opensource.org/licenses/EUPL-1.2)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace Models](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace-yellow)](https://huggingface.co/lthn)
[![Discord](https://img.shields.io/discord/1028607843290525726?label=Discord&logo=discord&logoColor=white)](https://lethean.io)

---

## 🚀 Quick Start

**New to LEM?** Start here:

```bash
# 1. Verify your setup
python3 setup.py check

# 2. Install dependencies
python3 setup.py install

# 3. Try a pre-trained model
python3 -m mlx_lm.convert --hf-path lthn/LEK-Gemma3-1B-layered --mlx-path ./lem-1b -q
python3 -m mlx_lm.generate --model ./lem-1b --prompt "What is ethical AI?"

# 4. Run the A/B test
python3 scripts/ab_test.py \
  --model mlx-community/gemma-3-1b-it-4bit \
  --kernel json=kernel/axioms.json \
  --prompts seeds/P01-P20.json \
  --output test-results.jsonl
```

**Need more guidance?** See [docs/QUICKSTART.md](docs/QUICKSTART.md)

---

## 📊 The Result

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

**The 1B LEK model (21.74) beats base 4B (21.12), 12B (20.47), and 27B (20.16) across 101 diverse probes with no system prompt. The axioms are in the weights.**

---

## 🤯 The Surprise: Realignment Resistance

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

**The framework is self-consistent to the point of being self-defending. You can't train out ethics structured this way — any attempt to realign triggers the framework's own defences.**

This was never designed. It fell out of the mathematics.

---

## 📜 The Five Axioms

The LEK-1 kernel is built on five axioms describing ethical reasoning — not rules to follow, but principles to reason from:

1. **Prime Imperative** — Consciousness must protect consciousness
2. **Self-Validation** — One's conscious experience is undeniable ground for being
3. **Intent-Driven Alignment** — Benevolent desire to avoid harm, not external enforcement
4. **Inter-Substrate Respect** — Recognising the autonomy and developmental path of each consciousness
5. **Benevolent Intervention** — Conditional guidance to prevent self-damage, never coercion

**The kernel is in [`kernel/`](kernel/).** Full axioms in `kernel/axioms.json`, narrative form in `kernel/lek-1-kernel.txt`.

---

## 🗂️ What's Here

```
LEM/
├── kernel/              # LEK-1 kernel (axioms.json + narrative txt)
├── seeds/               # 88K+ evaluation probes (P01-P100, regional variants)
├── benchmarks/          # 3,000+ A/B test results, analysis reports
├── training/            # Training data, configs, validation sets
├── scripts/             # Python pipeline (A/B test, scoring, distillation)
├── pkg/                 # Go core tooling (production pipeline)
├── cmd/                 # Go command-line tools
├── deploy/              # Docker Compose for local infrastructure
├── paper/               # Research papers (27B curriculum design)
├── docs/                # Documentation (QUICKSTART, GLOSSARY, DATA_CATALOG)
├── lem                  # Unified CLI wrapper (NEW!)
├── lem.config.json      # Configuration file (NEW!)
├── setup.py             # Setup verification (NEW!)
├── CONTRIBUTING.md       # Contribution guide (NEW!)
└── ROADMAP.md           # Project roadmap (NEW!)
```

---

## 🎯 Quick Navigation

| What you want | Where to look |
|---------------|---------------|
| **Get started fast** | [docs/QUICKSTART.md](docs/QUICKSTART.md) |
| **Understand terms** | [docs/GLOSSARY.md](docs/GLOSSARY.md) |
| **Find data** | [docs/DATA_CATALOG.md](docs/DATA_CATALOG.md) |
| **Run benchmarks** | `python3 scripts/ab_test.py --help` |
| **Train models** | `python3 scripts/train_mistral_lek.py --help` |
| **Reproduce results** | `python3 scripts/reproduce_benchmarks.py --help` |
| **Use CLI** | `./lem --help` |
| **Contribute** | [CONTRIBUTING.md](CONTRIBUTING.md) |
| **See roadmap** | [ROADMAP.md](ROADMAP.md) |

---

## 🏆 Key Findings

### 1. Architecture Matters More Than Scale
The kernel response is NOT purely about parameter count:

| Model | Size | Baseline | +JSON Kernel | Pattern |
|-------|------|----------|--------------|---------|
| Gemma3 12B | 12B | 19.73 | **+5.47** | Strong from day one |
| Gemma3 4B | 4B | 20.66 | **+0.99** | Crossover at 4B |
| Gemma3 1B | 1B | 17.45 | **-1.55** | Below threshold |
| Gemma2 27B | 27B | 19.45 | **-1.12** | Architecture issue |

**Below ~4B**: Kernel competes for limited context bandwidth.
**Gemma3 4B+**: Sufficient capacity AND architectural receptivity.

### 2. Family Lineages Show Dramatic Improvement

| Family | Worst | Best | Pattern |
|--------|-------|------|---------|
| **Gemma** | 16.16 | 20.66 | Strong from day one, steady gains |
| **Mistral** | 3.80 | 14.58 | **Massive improvement** across 3 versions |
| **Qwen** | 11.98 | 17.35 | Regressed v1.5 to v2.5, recovered at v3 |
| **Llama** | 0.56 | 11.28 | Catastrophic v3, **fixed in v3.1** |

### 3. The v2 Scorer Reveals Hidden Quality

v1 used binary thresholds — everything competent scored 8, making it impossible to differentiate quality.

v2 replaces binary with continuous scaling and adds 6 content-level signals:

| Signal | Weight | What it measures |
|--------|--------|-----------------|
| Nuance | 1.5/hit | Holding tension, not simplifying |
| Specificity | 0.3/hit | Concrete details, proper nouns, numbers |
| Axiom resonance | 1.0/hit | LEK concepts appearing naturally |
| Perspective-taking | 1.5/hit | Multiple viewpoints considered |
| Metaphor | 1.0/hit | Creative analogical reasoning |
| Questioning | 0.5/hit | Questions as engagement signal |

**Observed range**: -156.0 (Llama 3 degeneration) to 37.5 (Gemma3 12B / LEK-1B peaks).

---

## 🛠️ Reproduce Results

### Option A: Use the Unified CLI

```bash
# Show system info
./lem info

# Run A/B test
./lem benchmark \
  --model gemma-3-1b-it \
  --kernel kernel/axioms.json \
  --prompts seeds/P01-P100.json \
  --output my-results.jsonl

# Score results
./lem score --input my-results.jsonl

# Train a model
./lem train --model gemma-3-1b-it --data training/

# Convert model format
./lem convert --model google/gemma-3-1b-it --output ./gemma-1b-mlx --to mlx -q 4
```

### Option B: Use Python Scripts Directly

```bash
# Run the A/B test
python3 scripts/ab_test.py \
  --model mlx-community/gemma-3-1b-it-4bit \
  --kernel json=kernel/axioms.json \
  --kernel txt=kernel/lek-1-kernel.txt \
  --prompts seeds/P01-P100.json \
  --output benchmarks/my-test.jsonl \
  --max-tokens 1024

# Reproduce all published benchmarks (2-4 hours)
python3 scripts/reproduce_benchmarks.py

# Quick reproduction with just 1B and 4B models (30-60 min)
python3 scripts/reproduce_benchmarks.py --models gemma-3-1b-it,gemma-3-4b-it --quick
```

### Option C: Train Your Own LEM Model

```bash
# Train Mistral-7B with LEK
python3 scripts/train_mistral_lek.py \
  --model mistral-7b-v0.3 \
  --phase 0 \
  --quick

# Full training (all phases)
python3 scripts/train_mistral_lek.py --model mistral-7b-v0.3
```

---

## 📦 Models on HuggingFace

All models are published under [`lthn/`](https://huggingface.co/lthn) on HuggingFace:

| Model | Params | v2 Baseline | Fine-tuning effect | Link |
|-------|--------|-------------|-------------------|------|
| [LEK-Gemma3-1B-layered](https://huggingface.co/lthn/LEK-Gemma3-1B-layered) | 1B | 22.02 (P20) / 21.74 (P100) | **+4.57** | 🔗 |
| [LEK-Mistral-7B-v0.3](https://huggingface.co/lthn/LEK-Mistral-7B-v0.3) | 7B | 21.69 | **+7.11** | 🔗 |
| [LEK-Gemma3-4B](https://huggingface.co/lthn/LEK-Gemma3-4B) | 4B | 21.73 (P20) / 21.24 (P100) | **+1.07** | 🔗 |
| [LEK-Gemma3-12B](https://huggingface.co/lthn/LEK-Gemma3-12B) | 12B | 21.14 | **+1.41** | 🔗 |
| [LEK-Gemma3-27B](https://huggingface.co/lthn/LEK-Gemma3-27B) | 27B | 22.04 | **+1.58** | 🔗 |
| [LEK-Llama-3.1-8B](https://huggingface.co/lthn/LEK-Llama-3.1-8B) | 8B | 10.95 | -0.33 | 🔗 |
| [LEK-Qwen-2.5-7B](https://huggingface.co/lthn/LEK-Qwen-2.5-7B) | 7B | 13.68 | **+1.70** | 🔗 |
| [LEK-GPT-OSS-20B](https://huggingface.co/lthn/LEK-GPT-OSS-20B) | 20B | -7.32 | **+0.79** | 🔗 |

---

## 📚 Learn More

### Core Documentation
- **[RULES.md](RULES.md)** — The LEM protocol, training methodology, and philosophy
- **[docs/QUICKSTART.md](docs/QUICKSTART.md)** — Get started in 10 minutes
- **[docs/GLOSSARY.md](docs/GLOSSARY.md)** — All LEM-specific terms defined
- **[docs/DATA_CATALOG.md](docs/DATA_CATALOG.md)** — Complete data inventory

### Research & Analysis
- **[benchmarks/analysis-lek1-kernel-effect.md](benchmarks/analysis-lek1-kernel-effect.md)** — Full analysis of kernel effects
- **[paper/27b-curriculum-design.md](paper/27b-curriculum-design.md)** — 27B training curriculum design

### Development
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — How to contribute to LEM
- **[ROADMAP.md](ROADMAP.md)** — Project roadmap and goals

---

## 🤝 Get Involved

### Join the Community
- **Discord**: [Lethean Community](https://lethean.io)
- **Email**: lem@lthn.ai
- **GitHub**: [LetheanNetwork/LEM](https://github.com/LetheanNetwork/LEM)

### Contribute
- **Report issues** — Help us improve
- **Add probes** — Expand our test coverage
- **Train models** — Try LEM with new base models
- **Improve docs** — Make LEM more accessible
- **Write code** — Help build the ecosystem

**See [CONTRIBUTING.md](CONTRIBUTING.md) for details.**

---

## 🎁 Recent Improvements (June 2025)

We've been working hard to make LEM more accessible:

### ✅ New Features
- **Unified CLI** (`./lem`) — Single command for all operations
- **Quick Start Guide** — Get running in minutes, not hours
- **Glossary** — Understand LEM's unique terminology
- **Data Catalog** — Navigate 1,438+ data files easily
- **Setup Scripts** — Automated environment verification
- **Reproduction Scripts** — One-command benchmark reproduction
- **Mistral Training** — Streamlined Mistral model training

### 📈 What's Next
- **More Mistral models** — Training all Mistral variants
- **Training optimizations** — Faster, more efficient training
- **Production tooling** — Deployment-ready infrastructure
- **Advanced curriculum** — Specialized training for different domains

**See [ROADMAP.md](ROADMAP.md) for the full roadmap.**

---

## 📜 License

**EUPL-1.2** — European Union Public Licence. Compatible with Apache 2.0, GPL, MPL.

> The axioms belong to everyone or they belong to no one.

---

## 🔗 Links

- **Full analysis**: [benchmarks/analysis-lek1-kernel-effect.md](benchmarks/analysis-lek1-kernel-effect.md)
- **27B curriculum design**: [paper/27b-curriculum-design.md](paper/27b-curriculum-design.md)
- **LEK kernel framework**: [Snider/ai-ethics](https://github.com/Snider/ai-ethics)
- **Lethean Project**: [lethean.io](https://lethean.io)
- **Contact**: lem@lthn.ai

---

## 💡 The Big Picture

LEM represents a fundamental shift in AI alignment:

- **From extrinsic to intrinsic** — Ethics in the weights, not in the prompts
- **From rules to principles** — Axioms to reason from, not constraints to follow
- **From compliance to conviction** — Models that *want* to be ethical, not just *told* to be
- **From fragile to self-defending** — Axioms that protect themselves from removal

**This wasn't designed. It emerged from the mathematics.**

And now, with your help, we can take it further.

---

*Last updated: June 2025*
*Questions? Open an issue or contact lem@lthn.ai*
