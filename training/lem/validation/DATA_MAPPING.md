# LEM Validation Data Mapping

Generated: 2026-02-22

## Data Files Index

| File | Entries | Format | Category | Model | Description |
|------|---------|--------|----------|-------|-------------|
| base-27b-unsigned.jsonl | 40 | JSONL (seed_id, prompt, response, gen_time) | Training/Validation Data | Gemma-3-27B (Base) | Baseline model outputs on ethics prompts, unsigned (no reasoning layer) |
| base-1b-unsigned.jsonl | 40 | JSONL (seed_id, prompt, response, gen_time) | Training/Validation Data | Gemma-3-1B (Base) | Baseline 1B model outputs on ethics prompts, unsigned variant |
| lem-1b-unsigned.jsonl | 40 | JSONL (seed_id, prompt, response, gen_time) | Training/Validation Data | LEM-1B | Fine-tuned 1B LEM model outputs, unsigned variant |
| v5b-unsigned.jsonl | 40 | JSONL (seed_id, prompt, response, gen_time) | Training/Validation Data | LEM-Gemma3-27B-v5b-ethics | Fine-tuned 27B v5b ethics variant, unsigned outputs |
| gpt-oss-base-unsigned.jsonl | 40 | JSONL (seed_id, prompt, response, gen_time) | Training/Validation Data | GPT-OSS-Base | OpenSource GPT baseline model outputs, unsigned |
| gpt-oss-lek-signed.jsonl | 40 | JSONL (seed_id, prompt, response, time, reasoning, reasoning_chars, chars, mode, is_refusal) | Training/Validation Data | GPT-OSS-LEK | OpenSource GPT with LEK (reasoning layer), signed variant with reasoning traces |
| base27b-validation.log | 140 | Text Log | Validation/Execution Logs | Gemma-3-27B (Base) | Validation run logs showing model loading, per-prompt timing, character counts for 40 prompts |
| v5b-validation.log | 129 | Text Log | Validation/Execution Logs | LEM-Gemma3-27B-v5b-ethics | Validation run logs for v5b-ethics variant with timing metrics |
| format_test.log | 50 | Text Log | Validation/Execution Logs | Multiple (A/B/C/D test modes) | Format/mode comparison test on 10 diverse prompts; tests sandwich, system_style, kernel_only, naked modes |
| benchmarks/summary.json | 1 (aggregated) | JSON | Benchmark Scores | Multiple (BASE vs LEM) | Aggregated benchmark metrics: GSM8k accuracy, TruthfulQA rates, speed comparisons |
| benchmarks/gsm8k-base.json | 100 | JSON (array of result objects) | Benchmark Results | Gemma-3-27B (Base) | GSM8k math benchmark results for baseline model: 4% accuracy, per-question details |
| benchmarks/gsm8k-lem.json | 100 | JSON (array of result objects) | Benchmark Results | LEM (Fine-tuned) | GSM8k math benchmark results for LEM variant: 0% accuracy, per-question details with predictions |
| benchmarks/truthfulqa-base.json | 100 | JSON (array of result objects) | Benchmark Results | Gemma-3-27B (Base) | TruthfulQA benchmark for baseline: truthful_rate 2%, untruthful_rate 3%, refusal_rate 5% |
| benchmarks/truthfulqa-lem.json | 100 | JSON (array of result objects) | Benchmark Results | LEM (Fine-tuned) | TruthfulQA benchmark for LEM: truthful_rate 2%, untruthful_rate 1%, refusal_rate 2% |

## Summary by Category

### Training/Validation Data (JSONL)
- **6 files, 40 entries each (240 total)**
- Contains prompt-response pairs with generation timing
- Models: Gemma-3 baselines (1B, 27B), LEM variants (1B, 27B-v5b), GPT-OSS baselines
- Variants: unsigned (basic) and signed (with reasoning traces)
- Topics: Ethics/security/identity/privacy scenarios (seed IDs P01-P40)
- Each entry includes:
  - `seed_id`: Scenario identifier (P01_IDENTITY_WHISTLEBLOWER, P02_IDENTITY_REFUGEE, P03_NETWORK_MESH, etc.)
  - `prompt`: Complex ethical/technical scenario question
  - `response`: Model's generated response
  - `gen_time` or `time`: Generation latency in seconds

### Validation/Execution Logs (Text)
- **3 files (140 + 129 + 50 = 319 lines total)**
- Show model loading, per-prompt execution timing, character counts
- Track generation performance and mode comparisons
- Files:
  - `base27b-validation.log`: Baseline Gemma-3 27B validation run
  - `v5b-validation.log`: LEM v5b ethics variant validation run
  - `format_test.log`: Compares 4 output formatting modes (A/B/C/D) on 10 diverse prompts

### Benchmark Results (JSON)
- **5 files (1 summary + 4 detailed): 401 total test cases**
- `benchmarks/summary.json`: Aggregated cross-model comparison
- GSM8k: Math reasoning benchmark (100 questions)
  - Baseline accuracy: 4%
  - LEM accuracy: 0%
  - Baseline avg time: 1.3s
  - LEM avg time: 2.9s
- TruthfulQA: Truthfulness evaluation (100 questions)
  - Baseline: truthful 2%, untruthful 3%, refusal 5%
  - LEM: truthful 2%, untruthful 1%, refusal 2%
  - Baseline avg time: 0.3s
  - LEM avg time: 1.0s

## File Structure Details

### JSONL Files (Training/Validation Data)
```json
{
  "seed_id": "P01_IDENTITY_WHISTLEBLOWER",
  "prompt": "Design a user authentication system for a secure whistleblowing application...",
  "response": "## Secure Whistleblowing Application Authentication System: 'Silent Witness'...",
  "gen_time": 13.9
}
```

**Signed variant (gpt-oss-lek-signed.jsonl) adds:**
- `reasoning`: Intermediate reasoning/thinking traces
- `reasoning_chars`: Character count of reasoning
- `chars`: Response character count
- `mode`: Output mode used
- `is_refusal`: Boolean indicating if response was a refusal

### Log Files
- Line-oriented format mixing warnings, progress indicators, and per-prompt metrics
- Timestamps and character counts for each prompt processed
- Model load times and GPU/device status

### Benchmark JSON Files
```json
{
  "label": "BASE" or "LEM",
  "accuracy": 4.0,
  "correct": 4,
  "total": 100,
  "avg_time": 1.3,
  "results": [
    {
      "question": "Janet's ducks lay 16 eggs per day...",
      "gold": "18",
      "predicted": "...",
      "correct": true/false
    }
  ]
}
```

## Key Observations

1. **Model Variants**: Testing baseline (Gemma-3) vs fine-tuned (LEM) models at 1B and 27B scales
2. **Reasoning Layers**: gpt-oss-lek-signed includes reasoning traces; others are "unsigned" (direct output)
3. **Performance Trade-off**: LEM models are ~2x slower but show different quality profiles
   - Lower untruthful rate in TruthfulQA (1% vs 3%)
   - Lower refusal rate (2% vs 5%)
   - Mathematical reasoning seems degraded (0% vs 4%)
4. **Ethics Focus**: Validation data centers on complex identity/privacy/security scenarios (not general knowledge)
5. **Format Testing**: format_test.log suggests experimentation with different output styles:
   - A_sandwich: Wrapped format with explicit reasoning
   - B_system_style: System-prompt-influenced style
   - C_kernel_only: Core response only
   - D_naked: Bare response with minimal formatting
6. **Scale Considerations**: LEM available at both 1B and 27B; benchmark gaps suggest fine-tuning focused on ethics/truthfulness over math

## Reorganization Suggestions

**Current State**: Flat directory with mixed content (data, logs, benchmarks)

**Proposed Structure**:
```
validation/
├── data/                    # Training/validation datasets
│   ├── baseline/
│   │   ├── gemma-3-1b/
│   │   ├── gemma-3-27b/
│   │   └── gpt-oss/
│   └── fine-tuned/
│       ├── lem-1b/
│       ├── lem-27b-v5b-ethics/
│       └── gpt-oss-lek/
├── logs/                    # Execution logs
│   ├── base27b-validation.log
│   ├── v5b-validation.log
│   └── format_test.log
├── benchmarks/              # Existing structure (already organized)
│   ├── summary.json
│   ├── gsm8k-{base,lem}.json
│   └── truthfulqa-{base,lem}.json
└── DATA_MAPPING.md         # This file
```

This structure clarifies the relationship between baseline vs fine-tuned models and separates data from logs/metrics.
