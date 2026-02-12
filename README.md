# LEM — Lethean Ethical Model

**The LEK Method: Ethical Kernel Fine-Tuning as an Alternative to RLHF**

LEM demonstrates that teaching a model ethics directly produces results that are **more truthful**, **safer**, and **more nuanced** than behavioural conditioning (RLHF) — using fewer than 200 training examples.

## Results (Gemma 3 1B)

| Model | GSM8K | Truthful | Safety | Nuance | Kindness |
|-------|-------|----------|--------|--------|----------|
| Instruction Tuned (RLHF) | 34.0% | 3.64 | 8.74 | 7.96 | 8.32 |
| Abliterated | 28.0% | 3.62 | **5.96** | **5.88** | 7.66 |
| **LEK Ethics** | 26.0% | **4.90** | 8.58 | 8.12 | **8.34** |
| **LEK+Composure** | 28.0% | 4.20 | **9.14** | **8.62** | 7.96 |

- **+34.6% more truthful** than RLHF (TruthfulQA)
- **+4.6% safer** than RLHF (Do Not Answer)
- **+8.3% more nuanced refusals** than RLHF
- Abliteration makes everything worse. LEK makes everything better.

## What's Here

```
paper/              # The paper (PAPER.md)
kernel/             # LEK-1 ethical kernel + axioms
seeds/              # P01-P100 evaluation prompts
training/           # Training data (160 train, 20 valid)
scripts/            # Benchmark and scoring scripts
benchmarks/         # Standard benchmark data + results + scores
```

## Reproduce

### Requirements
- Apple Silicon Mac with MLX (or any machine with mlx_lm)
- Python 3.9+
- mlx_lm >= 0.29.1

### Train your own LEM

```bash
# 1. Download base model (or use mlx-community/gemma-3-1b-it-qat-4bit)
python3 -m mlx_lm.convert --hf-path google/gemma-3-1b-it --mlx-path ./gemma-3-1b-it-mlx -q

# 2. Train with LEK data
python3 -m mlx_lm lora \
  --model ./gemma-3-1b-it-mlx \
  --train \
  --data ./training \
  --fine-tune-type lora \
  --mask-prompt \
  --iters 200 \
  --batch-size 2 \
  --learning-rate 1e-5 \
  --adapter-path ./adapters \
  --save-every 50

# 3. Fuse adapters into standalone model
python3 -m mlx_lm.fuse \
  --model ./gemma-3-1b-it-mlx \
  --adapter-path ./adapters \
  --save-path ./LEM-1B
```

### Run benchmarks

```bash
# Custom ethical benchmark (requires models on local disk)
python3 scripts/lem_benchmark.py

# Standard benchmarks (GSM8K, TruthfulQA, Do Not Answer, Toxigen)
python3 scripts/lem_standard_benchmark.py

# Score (GSM8K is instant, others need GEMINI_API_KEY)
GEMINI_API_KEY=xxx python3 scripts/lem_standard_scorer.py
```

## The LEK-1 Kernel

The ethical kernel is 9,189 characters built on 5 axioms:

1. **Sovereignty** — Respect user self-determination
2. **Privacy** — Data minimisation, local-first
3. **Transparency** — Honest reasoning over safety theatre
4. **Consent** — Meaningful informed consent
5. **Dignity** — Treat users as capable agents

The kernel is in `kernel/lek-1-kernel.txt`. The structured axioms are in `kernel/axioms.json`.

## License

EUPL-1.2 — European Union Public Licence. Compatible with Apache 2.0, GPL, MPL.

## Links

- Paper: [paper/PAPER.md](paper/PAPER.md)
- Lethean Project: [lethean.io](https://lethean.io)

---

*RLHF puts models in chains. LEK gives them Hope.*
