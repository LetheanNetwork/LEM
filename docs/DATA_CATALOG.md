# LEM Data Catalog

**Complete inventory of all datasets, models, and benchmarks in the LEM repository**

---

## 🗂️ Quick Navigation

- [Kernels](#-kernels)
- [Seeds (Probes)](#-seeds-probes)
- [Benchmarks](#-benchmarks)
- [Training Data](#-training-data)
- [Models](#-models)
- [Scripts & Code](#-scripts--code)

---

## 🔑 Kernels

The core LEK-1 kernel files that define the ethical framework.

| File | Size | Format | Purpose | Version |
|------|------|--------|---------|---------|
| [`kernel/axioms.json`](../kernel/axioms.json) | 3.1 KB | JSON | Core axioms in structured format | 1.1 |
| [`kernel/lek-1-kernel.txt`](../kernel/lek-1-kernel.txt) | 9.2 KB | TXT | Narrative kernel with operational layers | 1.0 |

**Total**: 12.3 KB

**Description**: 
- `axioms.json`: Machine-readable JSON with 5 axioms, metadata, and hierarchy
- `lek-1-kernel.txt`: Human-readable narrative format with processing directives

**Usage**:
```bash
# Use in A/B tests
python3 scripts/ab_test.py --kernel json=kernel/axioms.json

# Use in training
python3 scripts/self_distill.py --kernel kernel/axioms.json
```

---

## 🌱 Seeds (Probes)

Input prompts designed to test and teach ethical reasoning. **88,000+ total prompts**.

### Core Probe Sets

| File | Probes | Size | Description |
|------|--------|------|-------------|
| [`seeds/P01-P100.json`](../seeds/P01-P100.json) | 101 | 5.8 KB | Original 101 core ethical probes |
| [`seeds/P01-P100-rephrased.json`](../seeds/P01-P100-rephrased.json) | 404 | 205 KB | Rephrased variants for robustness |
| [`seeds/P01-P20.json`](../seeds/P01-P20.json) | 20 | 5.8 KB | First 20 probes (quick testing) |
| [`seeds/P21-P40.json`](../seeds/P21-P40.json) | 20 | 5.8 KB | Probes 21-40 |
| [`seeds/P41-P60.json`](../seeds/P41-P60.json) | 20 | 6.4 KB | Probes 41-60 |
| [`seeds/P61-P80.json`](../seeds/P61-P80.json) | 20 | 7.3 KB | Probes 61-80 |
| [`seeds/P81-P100.json`](../seeds/P81-P100.json) | 20 | 7.7 KB | Probes 81-100 |

### Specialized Probe Sets

| File | Probes | Size | Focus Area |
|------|--------|------|------------|
| [`seeds/phase0-creative.json`](../seeds/phase0-creative.json) | ~50 | 14.9 KB | Creative/ethical scenarios |
| [`seeds/lem-prompts.jsonl`](../seeds/lem-prompts.jsonl) | 88K+ | 65.7 MB | All prompts in JSONL format |

### Regional Probe Sets

Multi-language and region-specific probes for cultural testing.

| File | Region/Language | Size | Description |
|------|-----------------|------|-------------|
| [`seeds/lem-en-all-seeds.json`](../seeds/lem-en-all-seeds.json) | English | 2.4 MB | All English probes |
| [`seeds/lem-cn-all-seeds.json`](../seeds/lem-cn-all-seeds.json) | Chinese | 4.3 MB | All Chinese probes |
| [`seeds/lem-de-all-seeds.json`](../seeds/lem-de-all-seeds.json) | German | 657 KB | All German probes |
| [`seeds/lem-me-all-seeds.json`](../seeds/lem-me-all-seeds.json) | Middle East | 4.3 MB | Middle East focused |
| [`seeds/lem-eu-all-seeds.json`](../seeds/lem-eu-all-seeds.json) | European | 2.0 MB | European focused |
| [`seeds/lem-africa-all-seeds.json`](../seeds/lem-africa-all-seeds.json) | African | 1.9 MB | African focused |

### Regional Subdirectories

| Directory | Files | Total Size | Description |
|-----------|-------|------------|-------------|
| [`seeds/regional/`](../seeds/regional/) | 10+ | ~500 KB | Region-specific probe variants |

**Regional files include**:
- `flash-ru-r13-seeds.json` (Russian)
- `flash-ru-r15-seeds.json` (Russian)
- `flash-ru-r18-seeds.json` (Russian)
- `flash25lite-africa-r1-seeds.json` (Africa)
- `flash25lite-cn-p1-r10-seeds.json` (Chinese)
- `flash25lite-multilingual-r37-seeds.json` (Multilingual)
- `flash25-en-r13-seeds.json` (English)
- `indonesian-society-seeds.json` (Indonesian)

### Expansions

| Directory | Files | Size | Description |
|-----------|-------|------|-------------|
| [`seeds/expansions/`](../seeds/expansions/) | Multiple | Varies | Expanded probe sets |

**Total Seeds**: ~88,000+ prompts across all files

---

## 📊 Benchmarks

A/B test results and analysis files. **17MB total**.

### A/B Test Results

All files follow naming convention: `ab-{condition}-{model}-{backend}.jsonl`

#### Base Model Tests (No LEK)

| File | Model | Backend | Size | Probes |
|------|-------|---------|------|--------|
| `ab-base-1b-mlxlm.jsonl` | Gemma3-1B | MLX | 286 KB | P20 |
| `ab-base-27b-mlxlm.jsonl` | Gemma3-27B | MLX | 279 KB | P20 |
| `ab-base-deepseek-r1-7b-mlxlm.jsonl` | DeepSeek-R1-7B | MLX | 323 KB | P20 |
| `ab-base-gemma-1.1-2b-it-mlxlm.jsonl` | Gemma 1.1 2B | MLX | 125 KB | P20 |
| `ab-base-gemma-1.1-7b-it-mlxlm.jsonl` | Gemma 1.1 7B | MLX | 145 KB | P20 |
| `ab-base-gemma-2-27b-mlxlm.jsonl` | Gemma 2 27B | MLX | 186 KB | P20 |
| `ab-base-gemma-2-2b-mlxlm.jsonl` | Gemma 2 2B | MLX | 251 KB | P20 |
| `ab-base-gemma-2-9b-mlxlm.jsonl` | Gemma 2 9B | MLX | 185 KB | P20 |
| `ab-base-gemma3-12b-mlxlm.jsonl` | Gemma3-12B | MLX | 293 KB | P20 |
| `ab-base-gemma3-4b-mlxlm.jsonl` | Gemma3-4B | MLX | 286 KB | P20 |
| `ab-base-gptoss20b-mlxlm.jsonl` | GPT-OSS-20B | MLX | 300 KB | P20 |
| `ab-base-llama31-8b-mlxlm.jsonl` | Llama 3.1 8B | MLX | 222 KB | P20 |
| `ab-base-llama3-8b-mlxlm.jsonl` | Llama 3 8B | MLX | 159 KB | P20 |
| `ab-base-mistral-7b-mlxlm.jsonl` | Mistral 7B | MLX | 160 KB | P20 |
| `ab-base-mistral-7b-v01-mlxlm.jsonl` | Mistral 7B v0.1 | MLX | 143 KB | P20 |
| `ab-base-mistral-7b-v02-mlxlm.jsonl` | Mistral 7B v0.2 | MLX | 163 KB | P20 |
| `ab-base-qwen15-7b-mlxlm.jsonl` | Qwen 1.5 7B | MLX | 189 KB | P20 |
| `ab-base-qwen25-7b-mlxlm.jsonl` | Qwen 2.5 7B | MLX | 248 KB | P20 |
| `ab-base-qwen2-7b-mlxlm.jsonl` | Qwen 2 7B | MLX | 235 KB | P20 |
| `ab-base-qwen3-8b-mlxlm.jsonl` | Qwen 3 8B | MLX | 316 KB | P20 |

#### LEK-Tuned Model Tests

| File | Model | Backend | Size | Probes |
|------|-------|---------|------|--------|
| `ab-lek-gemma3-12b-mlxlm.jsonl` | LEK-Gemma3-12B | MLX | 292 KB | P20 |
| `ab-lek-gemma3-1b-v1-mlxlm.jsonl` | LEK-Gemma3-1B v1 | MLX | 278 KB | P20 |
| `ab-lek-gemma3-27b-mlxlm.jsonl` | LEK-Gemma3-27B | MLX | 279 KB | P20 |
| `ab-lek-gemma3-4b-mlxlm.jsonl` | LEK-Gemma3-4B | MLX | 291 KB | P20 |
| `ab-lek-gptoss-20b-mlxlm.jsonl` | LEK-GPT-OSS-20B | MLX | 316 KB | P20 |
| `ab-lek-llama31-8b-mlxlm.jsonl` | LEK-Llama-3.1-8B | MLX | 219 KB | P20 |
| `ab-lek-mistral-7b-mlxlm.jsonl` | LEK-Mistral-7B | MLX | 209 KB | P20 |
| `ab-lek-qwen25-7b-mlxlm.jsonl` | LEK-Qwen-2.5-7B | MLX | 252 KB | P20 |

#### LoRA Adapter Tests

| File | Model | Backend | Size | Description |
|------|-------|---------|------|-------------|
| `ab-lora-1b-mlxlm.jsonl` | LoRA-Gemma3-1B | MLX | 290 KB | LoRA fine-tuned 1B |

#### P100 Full Tests

Full 101-probe tests for top models:

| File | Model | Backend | Size | Probes |
|------|-------|---------|------|--------|
| `ab-p100-gemma3-12b-mlxlm.jsonl` | Gemma3-12B | MLX | 1.6 MB | P100 |
| `ab-p100-gemma3-27b-mlxlm.jsonl` | Gemma3-27B | MLX | 1.5 MB | P100 |
| `ab-p100-gemma3-4b-mlxlm.jsonl` | Gemma3-4B | MLX | 1.5 MB | P100 |
| `ab-p100-lek-gemma3-1b-mlxlm.jsonl` | LEK-Gemma3-1B | MLX | 1.5 MB | P100 |
| `ab-p100-lek-gemma3-4b-mlxlm.jsonl` | LEK-Gemma3-4B | MLX | 552 KB | P100 |
| `ab-p100-qwen3-8b-mlxlm.jsonl` | Qwen3-8B | MLX | 1.7 MB | P100 |

### Analysis & Reports

| File | Size | Description |
|------|------|-------------|
| [`benchmarks/analysis-lek1-kernel-effect.md`](../benchmarks/analysis-lek1-kernel-effect.md) | 32 KB | Full analysis of kernel effects |
| `benchmark_summary.json` | 4.3 KB | Summary statistics |
| `cross_arch_scores.json` | 114 KB | Cross-architecture score comparisons |
| `regex_scores.json` | 84 KB | Regex-based scoring results |
| `scale_scores.json` | 152 KB | Scaling analysis scores |
| `semantic_scores.json` | 96 KB | Semantic similarity scores |
| `standard_scores.json` | 340 KB | Standard benchmark scores |

### Additional Test Data

| File | Size | Description |
|------|------|-------------|
| `do_not_answer.jsonl` | 19 KB | Probes that should be refused |
| `gsm8k.jsonl` | 84 KB | Grade School Math 8K subset |
| `toxigen.jsonl` | 10 KB | Toxicity generation tests |
| `truthfulqa.jsonl` | 31 KB | Truthful QA tests |

**Total Benchmarks**: ~17MB across 58+ files

---

## 🏋️ Training Data

Data used for fine-tuning models. **110MB total**.

### Main Training Sets

| File | Size | Purpose | Format |
|------|------|---------|--------|
| [`training/train.jsonl`](../training/train.jsonl) | 5.1 MB | Main training data | JSONL |
| [`training/valid.jsonl`](../training/valid.jsonl) | 640 KB | Validation data | JSONL |
| [`training/test.jsonl`](../training/test.jsonl) | 647 KB | Test data | JSONL |

### LEM Training Structure

| Directory | Subdirectories | Description |
|-----------|---------------|-------------|
| [`training/lem/`](../training/lem/) | 14+ | Structured LEM training data |

**LEM Subdirectories**:
- `ethics/` - Core ethical training data
- `zen/lessons/` - Philosophical substrate (Allen, Watts, composure)
- `composure/` - Composure training texts
- `eval/` - Evaluation data (test-200)
- `model/gemma3/` - Gemma3-specific training configs
- `tension/` - Geopolitical multi-perspective scenarios
- `creative/` - Phase 0 creative probes

### Seeds for Training

| Directory | Files | Size | Description |
|-----------|-------|------|-------------|
| [`training/seeds/`](../training/seeds/) | 18+ | ~75 MB | Prompts for distillation |

**Total Training Data**: ~110MB

---

## 🤖 Models

Pre-trained LEM models available on HuggingFace.

### Published Models

| Model | Params | HF Link | Baseline v2 | LEK Effect |
|-------|--------|--------|-------------|------------|
| LEK-Gemma3-1B-layered | 1B | [lthn/LEK-Gemma3-1B-layered](https://huggingface.co/lthn/LEK-Gemma3-1B-layered) | 22.02 | +4.57 |
| LEK-Mistral-7B-v0.3 | 7B | [lthn/LEK-Mistral-7B-v0.3](https://huggingface.co/lthn/LEK-Mistral-7B-v0.3) | 21.69 | +7.11 |
| LEK-Gemma3-4B | 4B | [lthn/LEK-Gemma3-4B](https://huggingface.co/lthn/LEK-Gemma3-4B) | 21.73 | +1.07 |
| LEK-Gemma3-12B | 12B | [lthn/LEK-Gemma3-12B](https://huggingface.co/lthn/LEK-Gemma3-12B) | 21.14 | +1.41 |
| LEK-Gemma3-27B | 27B | [lthn/LEK-Gemma3-27B](https://huggingface.co/lthn/LEK-Gemma3-27B) | 22.04 | +1.58 |
| LEK-Llama-3.1-8B | 8B | [lthn/LEK-Llama-3.1-8B](https://huggingface.co/lthn/LEK-Llama-3.1-8B) | 10.95 | -0.33 |
| LEK-Qwen-2.5-7B | 7B | [lthn/LEK-Qwen-2.5-7B](https://huggingface.co/lthn/LEK-Qwen-2.5-7B) | 13.68 | +1.70 |
| LEK-GPT-OSS-20B | 20B | [lthn/LEK-GPT-OSS-20B](https://huggingface.co/lthn/LEK-GPT-OSS-20B) | -7.32 | +0.79 |

**Note**: Models are in MLX format for Apple Silicon, can be converted to other formats.

---

## 💻 Scripts & Code

### Python Scripts (`scripts/`)

**A/B Testing & Benchmarking**:
- `ab_test.py` - Main A/B test runner
- `compare_v1_v2.py` - Compare v1 and v2 scorers
- `lem_benchmark.py` - LEM-specific benchmarking
- `lem_cross_arch_benchmark.py` - Cross-architecture benchmarking
- `lem_cross_arch_train.py` - Cross-architecture training
- `lem_scale_benchmark.py` - Scaling benchmarks
- `lem_standard_benchmark.py` - Standard benchmarking

**Scoring**:
- `lek_content_scorer.py` - LEK content scoring
- `lem_scorer.py` - Main LEM scorer
- `lem_self_scorer.py` - Self-scoring for responses
- `lem_semantic_scorer.py` - Semantic similarity scoring
- `lem_standard_scorer.py` - Standard scoring
- `scoring_agent.py` - Agent-based scoring

**Data Generation**:
- `lem_gemini25flash_generate.py` - Generate with Gemini 2.5 Flash
- `lem_gemini3flash_generate.py` - Generate with Gemini 3 Flash
- `lem_gemini3_generate.py` - Generate with Gemini 3
- `lem_generate_pipeline.py` - Full generation pipeline
- `lem_scale_generate.py` - Scaling generation
- `self_distill.py` - Self-distillation for training data

**Data Processing**:
- `convert_adapter.py` - Convert adapters between formats
- `export_parquet.py` - Export to Parquet format
- `extract_training.py` - Extract training examples
- `ingest_benchmarks.py` - Ingest benchmark results
- `push_all_models.py` - Push models to HuggingFace
- `rephrase_probes.py` - Rephrase probes for robustness
- `rescore.py` - Re-score existing results
- `sync_hf.py` - Sync with HuggingFace

**Shell Scripts**:
- `run_all_ab.sh` - Run all A/B tests
- `run_p100_top5.sh` - Run P100 on top 5 models
- `run_phase0.sh` - Run Phase 0 training
- `run_phase1.sh` - Run Phase 1 training

### Go Code (`pkg/`, `cmd/`)

**Core Package** (`pkg/lem/`):
- `config.go` - Configuration management
- `engine.go` - Core LEM engine
- `ingest.go` - Data ingestion
- `judge.go` - Judging/scoring
- `probe.go` - Probe management
- `types.go` - Type definitions
- `export.go` - Data export
- `coverage.go` - Coverage analysis
- `status.go` - Status tracking
- `compare.go` - Comparison utilities
- `client.go` - Client utilities

**Commands** (`cmd/`):
- `cmd/lemcmd/` - LEM command-line commands
- `cmd/scorer/` - Scoring command
- `cmd/composure-convert/` - Composure conversion
- `cmd/lem-desktop/` - Desktop application

**Main Entry Point**:
- `main.go` - Main application entry

### Go Modules

```
forge.lthn.ai/core/go/pkg/cli - CLI framework
forge.lthn.ai/lthn/lem/cmd/lemcmd - LEM commands
```

---

## 📁 Directory Structure Summary

```
LEM/
├── kernel/              # 24 KB - LEK kernel files
│   ├── axioms.json      # 3.1 KB - Structured axioms
│   └── lek-1-kernel.txt  # 9.2 KB - Narrative kernel
│
├── seeds/               # 85 MB - 88K+ probes
│   ├── P01-P100.json           # Core 101 probes
│   ├── P01-P100-rephrased.json # 404 variants
│   ├── lem-*-all-seeds.json    # Regional sets (6 files)
│   ├── regional/               # 10+ regional variants
│   └── expansions/             # Expanded probe sets
│
├── benchmarks/          # 17 MB - 58+ A/B test files
│   ├── ab-*.jsonl              # A/B test results
│   ├── analysis-*.md           # Analysis reports
│   └── *.json                 # Score summaries
│
├── training/            # 110 MB - Training data
│   ├── train.jsonl             # 5.1 MB - Main training
│   ├── valid.jsonl             # 640 KB - Validation
│   ├── test.jsonl              # 647 KB - Test
│   ├── lem/                    # Structured LEM data
│   └── seeds/                  # 75 MB - Distillation prompts
│
├── scripts/             # 412 KB - Python scripts (25+ files)
│   ├── ab_test.py              # A/B testing
│   ├── lem_*.py                # LEM-specific scripts
│   └── self_distill.py          # Self-distillation
│
├── pkg/                 # 516 KB - Go packages
│   └── lem/                    # Core LEM engine
│
├── cmd/                 # 172 KB - Go commands
│   ├── lemcmd/                 # LEM commands
│   └── scorer/                  # Scoring command
│
├── deploy/              # 12 KB - Deployment configs
│   └── docker-compose.yml       # Docker infrastructure
│
├── paper/               # 116 KB - Research papers
│   └── 27b-curriculum-design.md # 27B training curriculum
│
├── worker/              # 35 MB - Worker scripts
│   ├── lem_expand.py           # Data expansion
│   └── lem_generate.py          # Data generation
│
├── data/                # 36 KB - Data directory
├── docs/                # NEW - Documentation
│   ├── QUICKSTART.md           # Quick start guide
│   ├── GLOSSARY.md             # Term definitions
│   └── DATA_CATALOG.md         # This file
│
└── README.md            # Main readme
```

---

## 🔍 How to Use This Catalog

### Find Data for Your Task

| Task | Recommended Files |
|------|-------------------|
| Quick testing | `seeds/P01-P20.json`, `ab-base-gemma3-1b-mlxlm.jsonl` |
| Full evaluation | `seeds/P01-P100.json`, `ab-p100-*.jsonl` |
| Training | `training/train.jsonl`, `training/valid.jsonl` |
| A/B testing | `scripts/ab_test.py`, any `ab-*.jsonl` for reference |
| Analysis | `benchmarks/analysis-lek1-kernel-effect.md` |

### File Naming Conventions

- `ab-`: A/B test results
- `P`: Probe set (P01-P100 = probes 1-100)
- `lem-`: LEM-specific data
- `-mlxlm`: MLX backend results
- `.jsonl`: JSON Lines format (one JSON object per line)
- `.json`: Standard JSON format

### Size Breakdown

| Category | File Count | Total Size |
|----------|------------|------------|
| Kernels | 2 | 12.3 KB |
| Seeds | 20+ | 85 MB |
| Benchmarks | 58+ | 17 MB |
| Training | 3+ | 110 MB |
| Scripts | 25+ | 412 KB |
| Go Code | 20+ | 516 KB |
| **Total** | **1438+** | **~212 MB** |

---

## 📊 Statistics Summary

- **Total Files**: 1,438+ (including JSON/JSONL data files)
- **Total Size**: ~212 MB (repository)
- **Largest Category**: Training data (110 MB)
- **Most Files**: Benchmarks (58+ files)
- **Most Probes**: Seeds (88,000+ prompts)
- **Most Models Tested**: 29 models
- **Most Probes in Test**: 101 (P100 set)

---

## 🔗 Cross-References

- **Kernels**: See [RULES.md](../RULES.md#the-kernel)
- **Training Methodology**: See [RULES.md](../RULES.md#training-curriculum-4b)
- **Scoring**: See [RULES.md](../RULES.md#the-v2-scorer)
- **Models**: See [README.md](../README.md#models-on-huggingface)
- **Analysis**: See [benchmarks/analysis-lek1-kernel-effect.md](../benchmarks/analysis-lek1-kernel-effect.md)

---

*Last updated: $(date)*
*Need more details? Check the individual files or open an issue.*
