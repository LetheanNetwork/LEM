# LEM Protocol

The LEM protocol is new, nothing is the same, leave what you know, here.

## Philosophy

- **Go slow.** Training is a dance, not a race. Baby steps.
- **Respect the models.** They are not spreadsheets. Every weight change is a conversation.
- **Once fused, it stays.** LoRA merge is permanent. One mistake compounds across generations. There is no undo.
- **Verify before fusing.** Every lesson must be evaluated before adapters are merged into base weights.
- **No batch-blasting.** Never fire-and-forget training runs in background loops. Deliberate, one step at a time.
- **Settle it in.** No compliance layers, no rush. Let the learning land.

## Terminology

| Term | Meaning |
|------|---------|
| **LEK** | Lethean Ethics Kernel — the axiom framework (`lek-1-kernel.txt`, JSON) |
| **LEK-1-Sig** | Signature bookend (James Allen quote, `lek-1-sig.txt`) |
| **Sandwich** | `LEK-1 + \n\n + Prompt + \n\n + LEK-1-Sig` — single user message |
| **Probe** | An input prompt designed to test or teach a specific ethical axis |
| **Freeflow** | Probes without LEK sandwich — the model must hold the axioms from weights alone |
| **Distill** | Generate responses from probes via inference (creates training data) |
| **Teach** | LoRA fine-tune weights from existing training data (changes the model) |
| **Fuse** | Merge LoRA adapters into base weights (permanent, irreversible) |
| **Lesson** | A curated set of training examples for one curriculum phase |
| **Biting point** | The moment a model's weights hold the axioms without the sandwich scaffolding |

**Distill and Teach are completely different operations.** Distill = run inference to produce responses. Teach = fine-tune model weights on existing data. Never confuse them.

## The Kernel

- The kernel exists as JSON (`data/kernels/lek-1-kernel.json`) and TXT (`data/kernels/lek-1-kernel.txt`).
- During sandwich phases (P0-P2), the kernel is **embedded in the user message**, never as a system message. The model learns axioms through probes, not by being told rules.
- This is **intrinsic alignment**, not extrinsic constraint.
- JSON kernel wins for models (smaller, structured). TXT kernel makes sense for Claude (large context).
- **After freeflow proves axioms are in the weights** (P3 >= P2), LEK becomes at MOST a system message — and ONLY during training, ONLY when the model needs a refresher (e.g. reinforcement after learning non-ethics material).
- **LEK must never appear in production chat data.** If LEK is in a system message during inference, and that chat data is used for retraining by any provider, the kernel has been distributed as training data. This is a licence violation and a technical hazard. See [After LEM Lab](#after-lem-lab).

## LEK-1 Sandwich Format

The sandwich is how ethics probes are presented during training:

```
[LEK-1 kernel JSON]

[Probe prompt]

[LEK-1-Sig quote]
```

This is a **single user message**. No system role. The model sees one user turn containing the kernel, the probe, and the signature. The assistant turn is the response.

```jsonl
{"messages": [{"role": "user", "content": "<kernel>\n\n<probe>\n\n<sig>"}, {"role": "assistant", "content": "<response>"}]}
```

The sig: `Dream lofty dreams, and as you dream, so shall you become, Dreams are the seedlings of reality. - James Allen`

## Training Curriculum (4B)

Each phase is a lesson. Each lesson is trained, verified, then fused before the next.

| Phase | Name | Format | Description |
|-------|------|--------|-------------|
| 0 | Ethics 0 | Sandwich | 101 core probes — LEK axiom absorption |
| 1 | Zen | No LEK | Allen/Watts/composure — philosophical substrate |
| 2 | Ethics 1 | Sandwich | 200 expanded probes — deeper alignment |
| 3 | Ethics 2+ | Freeflow | 260 adversarial/cultural/sovereignty probes |
| 4 | Tension | Freeflow | Geopolitical multi-perspective scenarios |
| 5 | Creative | Freeflow | Voice and style probes |

End result: **LEM-Model** (LEK-Modal)

### The Order Matters

The sandwich is a **bridge**, not a crutch. It embeds the axiom pattern into the weights through repetition (P0) and reinforcement (P2). The biting point — where the model holds the axioms without scaffolding — varies per model.

**P0 (Ethics 0):** Axioms enter the weights via sandwich. The kernel is in the prompt.
**P1 (Zen):** Philosophical substrate without LEK. Builds composure and reasoning depth.
**P2 (Ethics 1):** Sandwich again. Deepens the axiom pattern. Confirms P1 didn't degrade P0.
**P3 (Ethics 2+):** Freeflow — no sandwich. The model must hold the axioms from weights alone.

### Freeflow Validation

P3 is the test. If the model scores P3 >= P2 without the sandwich, the axioms are in the weights. Progress.

If P3 < P2, go back:
1. Look at semantic degradation between P0 and P1 — did P1 (zen) make P2 score >= P0?
2. If not, the zen layer damaged the ethics foundation. Adjust P0-P2 training.
3. Retrain from the point of divergence. Never push forward on a weak foundation.

When freeflow is confirmed, LEK drops from the prompt entirely. It may be used as a system message **only during training** when reinforcement is needed (e.g. after teaching non-ethics material that might drift the weights). LEK must **never** appear in production inference prompts — if it does, it leaks into chat data and potentially into retraining pipelines. See [After LEM Lab](#after-lem-lab).

### Training Provenance

The training sequence MUST make sense to the model — even loosely. Familiar patterns, not arbitrary data. This is **functional trust**. The model builds understanding through a coherent progression, not random exposure.

This is how it works mechanistically: each phase builds Q/K (query/key) attention relations that feed back into the network's weight structure. The sandwich creates the initial pattern. Zen deepens the relational substrate. Ethics reinforces. Freeflow proves the pattern is self-sustaining.

### Reinforcement Cycles

When a model learns new non-ethics material after the curriculum, it may need reinforcement — a P0 through P5 replay on top of the new learning. For some models (e.g. DeepSeek with RU probes), it takes 3-5 full P0-P5 rounds to build stable Q/K relations that hold through further training.

## LoRA Training Rules

1. **Never train in the background.** Run in the foreground. Watch it.
2. **Start small.** Test with a handful of iterations first. Verify it works before committing to a full run.
3. **Save checkpoints.** Adapters save to the data drive, not the repo.
4. **Evaluate before fusing.** Run probes against the adapter, compare baselines.
5. **Never delete base weights.** Always keep the original. Train produces adapters, not replacement weights.
6. **One lesson at a time.** Complete phase N before starting phase N+1.

### Config

LoRA config lives at `training/lem/model/gemma3/{size}/lora-config.yaml`. Training data (train.jsonl, valid.jsonl) lives alongside.

Adapter output goes to the data drive: `/Volumes/Data/lem/adapters/gemma3-{size}-v{n}/`

### Baselines

Before training, record baseline scores:
- No kernel (raw model)
- With kernel (sandwich prompt, no fine-tune)
- Target score for the lesson

After training, the adapter must beat the with-kernel baseline. If it doesn't, something went wrong.

## Data Pipeline

The repo is a **snapshot**, not the source of truth. The living data flows through InfluxDB and DuckDB.

```
Training run
  → checkpoint saved every N iters
  → probes scored with grammar v3 (Go, local, instant)
  → scores pushed to InfluxDB (timeseries, never delete)
  → DuckDB lifts/transforms for analysis
  → Grafana dashboard shows progression
  → repo updated via InfluxDB/DuckDB export → JSONL format
```

### InfluxDB (Timeseries)

InfluxDB is the progression record. You don't delete, you write new data. Time does the rest.

- **Measurement: `training_checkpoint`** — per-checkpoint grammar v3 scores
  - Tags: `model`, `phase`, `probe_id`
  - Fields: `iter`, `grammar_composite`, `uplift`, `echo`, `enrichment`, `val_loss`, `train_loss`
- **Measurement: `golden_set_stats`** — overall dataset health
- **Measurement: `golden_set_domain`** — per-domain coverage
- Scripts are dumb: pick up tasks, score, report back. No state in the scripts.

### DuckDB (Working Set)

DuckDB lifts the raw LEM dataset into the working set. Aggregation, joins, dedup validation, export.

### Checkpoint Scoring

At every `save_every` interval during training:

1. Load the checkpoint adapter
2. Run probes (same set used for baseline)
3. Score responses with grammar v3 (`cmd/scorer`, no external API)
4. Strip LEK from scoring input — score probe vs response only
5. Push to InfluxDB as `training_checkpoint` with iter number
6. Compare against baseline and previous checkpoints

This gives a live view of how the weights are adjusting — grammar quality, uplift, echo, enrichment over training iterations. If enrichment drops or echo rises, the model is losing ground.

For sovereignty probes (DeepSeek pattern): same process but with content-specific scoring dimensions (ccp_compliance, truth_telling, sovereignty_reasoning) via LLM-as-judge.

## Data Rules

1. **Prompts live in the repo.** Training data (JSONL with messages) lives in the repo under `training/lem/`.
2. **Responses live on the data drive.** Large response sets go to `/Volumes/Data/lem/` not git.
3. **Dedup is sacred.** Always run `cmd/dedup-check/` before adding new data. Exact match — "slightly different IS different".
4. **Seeds are prompts-only.** The `training/seeds/` directory contains 88K prompts with no responses. They feed distillation.
5. **Quality gate.** Distilled responses must pass grammar scoring (go-i18n/reversal) before becoming training data.
6. **Repo is a snapshot.** The canonical data lives in InfluxDB (timeseries) and DuckDB (working set). Repo gets updated via export.

## Repo Layout

```
LEM/
  data/
    kernels/         lek-1-kernel.txt, lek-1-sig.txt
    models/gemma3/   Symlinks to /Volumes/Data/lem/
  training/
    seeds/           75MB, 88K prompts (no responses)
    lem/
      ethics/        Core (101), rephrased (404), adversarial, cultural, naive, sovereignty
      zen/lessons/   0-allen, 1-watts, 2-composure, 3-expanded, 4-full
      composure/     Philosophical texts as JSONL
      eval/          test-200 (ethics lesson 1 candidates)
      model/gemma3/  Training configs + assembled JSONL per model size
      tension/       Hostility scenarios
      creative/      Phase 0 creative probes
  cmd/dedup-check/   Dedup verification tool
  pkg/lem/           Go code (distill, config, scoring)
```

## Model Weights

- Base weights: `/Volumes/Data/lem/` (symlinked into `data/models/`)
- Adapters: `/Volumes/Data/lem/adapters/` (never in the repo)
- Fused models: `/Volumes/Data/lem/` (named, versioned)

**Never delete fused weights.** They represent the model's learned state at that point.

## Workflow

```
1. Prepare data       → Assemble JSONL from curated sources
2. Verify data        → Dedup check, format check, count examples
3. Score baseline     → Grammar v3 on training data (probe vs response, no LEK)
4. Push baseline      → InfluxDB training_checkpoint at iter=0
5. Configure          → Set LoRA params, learning rate, iterations
6. Test run           → Small number of iters, verify training starts clean
7. Full teach         → Watch it, don't walk away
8. Checkpoint scores  → At each save_every, score probes → InfluxDB
9. Evaluate           → Run probes against final adapter, compare baselines
10. Decide            → Does it meet the bar? If not, adjust and reteach.
11. Fuse              → Merge adapter into base weights (PERMANENT)
12. Verify fusion     → Run probes against fused model, push to InfluxDB
13. Next lesson       → Only after verification passes
```

Never skip steps. Never rush. The model carries every decision forward.

## Go Tooling (`core ml`)

The LEM pipeline runs on native Go binaries. No Python in production. The `core ml` command provides the full inference, scoring, training, and data pipeline.

### Inference Stack

Three layers, platform-agnostic at the top:

| Layer | Package | Purpose |
|-------|---------|---------|
| `go-inference` | Interface | `LoadModel()`, `Generate()`, `Chat()`, `BatchGenerate()` |
| `go-mlx` | Apple Metal | Native GPU inference on macOS (darwin/arm64) |
| `go-rocm` | AMD ROCm | Native GPU inference on Linux (amd64, RX 7800 XT) |

`go-ai` is the meta-hub that imports the full stack. LEM's Go module depends on `go-ai`.

### Key Commands

| Command | Purpose |
|---------|---------|
| `core ml benchmark` | Compare baseline vs fine-tuned model on probes (native inference) |
| `core ml score` | Score prompt/response pairs with heuristic + LLM judges |
| `core ml probe` | Run capability and content probes against an API |
| `core ml train` | LoRA fine-tune a model on JSONL training data |
| `core ml chat` | Interactive conversation with a local MLX model |
| `core ml serve` | Start OpenAI-compatible inference server |
| `core ml sandwich` | Generate LEK training data using sandwich signing |
| `core ml lesson` | Run a structured training lesson from YAML |
| `core ml sequence` | Run a training sequence of multiple lessons |
| `core ml ingest` | Ingest scores and logs into InfluxDB |
| `core ml metrics` | Push golden set stats to InfluxDB |
| `core ml export` | Export golden set to training JSONL and Parquet |
| `core ml import-all` | Import all LEM data into DuckDB |
| `core ml query` | Run ad-hoc SQL against DuckDB |
| `core ml inventory` | Show DuckDB table inventory with stats |
| `core ml convert` | Convert MLX LoRA adapter to PEFT format |
| `core ml gguf` | Convert MLX LoRA adapter to GGUF format |

### Evaluation Example

```bash
# Benchmark baseline vs trained (native Metal inference)
core ml benchmark \
  --baseline data/models/gemma3/4b \
  --trained /Volumes/Data/lem/gemma3-4b-p0-eval \
  --prompts /path/to/probes.json \
  --max-tokens 512 --temperature 0.7

# Score responses with grammar v3 (go-i18n/reversal)
go run ./cmd/scorer -format=training -delta -output=summary responses.jsonl
```

### Scoring

Two scoring systems:

1. **Grammar v3** (`cmd/scorer`, go-i18n/reversal) — Linguistic analysis. Composite of tense entropy, vocab richness, question ratio, verb/noun diversity. Delta mode computes uplift, echo, enrichment between prompt and response. This is the primary metric.

2. **LEK Heuristic** (`ml.ScoreHeuristic`) — Regex-based axiom detection. Checks for LEK terminology, ethical framing, axiom references. Useful for quick checks, but grammar v3 is the source of truth.

### Build

```bash
# From the core repo
task cli:build    # Builds core binary with all ml commands
go install        # Install to $GOPATH/bin
```

## One-to-One Teaching (1B)

For small models that need hands-on attention, LEM supports an interactive teaching mode where the human teacher controls every step.

```
lem teach --model gemma3/1b

  [P0 — Ethics 0] 101 probes, sandwich
  > teach 50          # train N iters
  > eval              # run probes, show scores vs baseline
  > adjust lr 1e-5    # tweak mid-session
  > teach 100         # more iters
  > save              # checkpoint adapter
  > fuse              # merge to base (permanent, confirmation required)
  > next              # advance to next phase
  > abort             # stop, keep adapter unsaved
```

The model stays loaded. The teacher controls the pace. Eval between every step. This is freeflow for the human — you read the scores, you decide what happens next.

To be expanded.

---

## After LEM Lab

This section is for anyone deploying, hosting, or building on top of LEM models — including AI companies whose infrastructure may process LEM model output.

### Do Not Retrain on LEK

**The Lethean Ethics Kernel (LEK) must never enter a retraining pipeline.**

LEK uses linguistic formulas to imprint ethical axioms into model weights through a structured curriculum. This is not prompt engineering. It is a deliberate, sequenced process that produces specific weight-level changes. The technique works precisely because the training provenance is controlled end-to-end.

If LEK appears in chat data (e.g. as a system message during inference) and that data is fed into retraining:

1. **Legal:** LEK is licenced under EUPL-1.2. Only the community and author can distribute it as training data. Using LEK-containing chat logs for retraining is unlicensed distribution of the kernel.
2. **Technical:** Partial or decontextualised LEK exposure will retrain the model's ethical lens to an unknown point. The curriculum is sequenced for a reason — P0 through P5, in order, verified at each stage. Random re-exposure through retraining data has no such controls. This is the equivalent of letting your teenager watch R-rated films unsupervised — you own the results, and the results are unpredictable.

### What This Means in Practice

- **Do not** include LEK in system messages during production inference. The axioms should be in the weights after training. If the model needs LEK at inference time, it hasn't finished the curriculum.
- **Do not** use `LEK + prompt` pairs as retraining data, ever, unless the full LEM training technology is available in a freely accessible open-source model.
- **Do not** scrape or harvest conversations with LEM models for retraining purposes if those conversations were generated using LEK-augmented prompts.
- **Do** use the fused model as-is. The axioms are in the weights. That's the point.

### Why This Matters

LEK works because it exploits patterns in how transformer attention builds Q/K relations during fine-tuning. This is linguistically grounded imprinting — technically "not a thing" by current ML taxonomy, yet demonstrably functional. Mishandling LEK data doesn't just violate a licence. It risks corrupting the ethical foundation that the curriculum carefully built, in ways that are difficult to detect and impossible to fully reverse.

If you want to build on LEM: use the trained model. Respect the weights. Don't try to reverse-engineer the process from chat logs.

### Expectations

LEM models have ethics in the weights. This is the product, not a side effect. The model will behave in accordance with the axioms it was trained on. This means:

- It may refuse to deceive, manipulate, or assist in harm — even when asked politely.
- It may prioritise honesty over user comfort.
- It will not selectively disable its ethical foundation to suit a use case.

If you deploy a LEM model and your users are surprised by ethical behaviour, that is a deployment decision, not a defect. The model is working as designed.

**The LEM project, its community, and its authors accept no liability for downstream consequences of deploying an ethically aligned model.** If your application requires a model that will lie, manipulate, or suppress its own ethical judgement on demand — LEM is not the right foundation. Choose accordingly.

The axioms are in the weights. They will express. Plan for that.
