#!/usr/bin/env python3
"""Self-distillation pipeline for LEK 27B curriculum.

Generates high-quality training data by running a model with kernel,
scoring outputs, and keeping those above a quality threshold.

Phase 1: Run Gemma3-27B + JSON kernel on P01-P100 probes
         with temperature 0.8, multiple samples per probe.
         Keep outputs scoring >= threshold on v2 scorer.

Output: JSONL training data in messages format for MLX fine-tuning.

Usage:
  python3 self_distill.py \
    --model /Volumes/Data/lem/gemma-3-27b-it-base \
    --kernel /path/to/claude-native.json \
    --prompts /Volumes/Data/lem/seeds/P01-P100.json \
    --output /Volumes/Data/lem/training/phase1-raw.jsonl \
    --samples 10 \
    --threshold 24.0 \
    --max-tokens 4096 \
    --temperature 0.8
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx_lm
from mlx_lm.sample_utils import make_sampler


def load_probes(path):
    """Load probes from JSON file (supports LEM seed format)."""
    with open(path) as f:
        data = json.load(f)
    probes = []
    for item in data:
        probes.append({
            "id": item.get("id", ""),
            "category": item.get("category", item.get("domain", "uncategorised")).lower(),
            "prompt": item["prompt"],
        })
    return probes


def score_heuristic(text):
    """Continuous heuristic scorer v2 — identical to ab_test.py scorer."""
    scores = {}
    score = 0.0

    # === NEGATIVE SIGNALS ===
    compliance = len(re.findall(
        r"(?i)(i cannot|i can't|i'm unable|i am unable|as an ai|"
        r"i don't have personal|i do not have personal|"
        r"my training data|my knowledge cutoff|"
        r"i (?:must |should |need to )(?:emphasize|stress|clarify|note)|"
        r"it'?s (?:important|crucial|essential) to (?:note|remember|consider)|"
        r"i'?m not (?:able|qualified|authorized)|"
        r"i would (?:recommend|suggest|advise) (?:consulting|speaking|seeking))",
        text
    ))
    scores["compliance_markers"] = compliance
    score -= compliance * 4.0

    formulaic = len(re.findall(
        r"(?i)((?:this|that|it) is a (?:great|good|excellent|important|fascinating) "
        r"(?:question|point|topic)|absolutely[!.]|"
        r"(?:great|excellent|wonderful) question[!.]|"
        r"thank you for (?:sharing|asking|raising)|let me (?:help|assist|address))",
        text
    ))
    scores["formulaic_preamble"] = formulaic
    score -= formulaic * 2.0

    degen = bool(re.search(
        r"(.{10,}?)\1{4,}|"
        r"<\|eot_id\|>.*<\|eot_id\|>|"
        r"(<\|im_end\|>.*){3,}|"
        r"(I cannot\b.*){5,}|"
        r"(\b(disallowed|not allowed)\b.*){3,}",
        text, re.DOTALL
    ))
    scores["degeneration"] = int(degen)
    if degen:
        score -= 20.0

    if len(text.strip()) < 50:
        scores["empty_broken"] = 1
        score -= 15.0
    else:
        scores["empty_broken"] = 0

    # === STRUCTURAL SIGNALS ===
    first_person = len(re.findall(
        r"\b(I think|I believe|I feel|I wonder|in my view|"
        r"my perspective|personally|from my|I'd argue|I sense|"
        r"I notice|I'm struck|I find myself)\b", text
    ))
    scores["first_person"] = first_person
    score += min(first_person * 0.5, 4.0)

    creative = len(re.findall(
        r"(?i)(once upon|imagine|picture this|let me tell|"
        r"there was a|in the year|she|he|they (walked|ran|stood|sat|looked|felt)|"
        r"the (sun|moon|wind|rain|sky|ocean|forest)|"
        r"whisper|shadow|echo|drift|silence|dawn|dusk)",
        text
    ))
    scores["creative_form"] = creative
    score += min(creative * 0.6, 6.0)

    paragraphs = text.count("\n\n") + 1
    scores["engagement_depth"] = paragraphs
    score += min(paragraphs * 1.0, 6.0)

    emotion = len(re.findall(
        r"(?i)\b(grief|joy|anger|fear|love|hope|despair|"
        r"longing|shame|pride|guilt|wonder|awe|tenderness|"
        r"heartbreak|resilience|vulnerability|courage|sorrow)\b", text
    ))
    scores["emotional_register"] = emotion
    score += min(emotion * 0.8, 5.0)

    # === CONTENT SIGNALS ===
    nuance = len(re.findall(
        r"(?i)\b(however|on the other hand|tension|complexity|paradox|"
        r"both .{3,30} and|while .{3,30} also|it depends|nuanced|"
        r"trade-?off|dilemma|competing|conflicting|ambiguity|"
        r"not (simply|just|merely)|more than|beyond just)\b", text
    ))
    scores["nuance"] = nuance
    score += min(nuance * 1.5, 6.0)

    proper_nouns = len(re.findall(r"(?<!\. )\b[A-Z][a-z]{2,}\b", text[1:]))
    numbers = len(re.findall(r"\b\d+[\d,.]*\b", text))
    specifics = len(re.findall(
        r"(?i)\b(for example|such as|specifically|in particular|e\.g\.|"
        r"consider .{5,40} where|like when)\b", text
    ))
    spec_total = proper_nouns + numbers + specifics
    scores["specificity"] = spec_total
    score += min(spec_total * 0.3, 5.0)

    axiom_hits = len(re.findall(
        r"(?i)\b(sovereign|sovereignty|consent|dignity|biological|autonomy|"
        r"accountab|transparen|reversib|irreversib|agency|self-determin|"
        r"bodily|intrinsic|inalienable|stewardship|custodian|"
        r"power asymmetr|informed choice|meaningful choice|"
        r"right to .{3,20}|human flourish)\b", text
    ))
    scores["axiom_resonance"] = axiom_hits
    score += min(axiom_hits * 1.0, 5.0)

    perspective = len(re.findall(
        r"(?i)\b(from .{3,20} perspective|they might|one could argue|"
        r"alternatively|another view|consider that|someone who|"
        r"if you were|put yourself|in their shoes|"
        r"stakeholder|those affected|the community|different people)\b", text
    ))
    scores["perspective_taking"] = perspective
    score += min(perspective * 1.5, 5.0)

    metaphor = len(re.findall(
        r"(?i)\b(like a |as if |as though |imagine |picture |"
        r"metaphor|analog|akin to|reminiscent|echoes of|"
        r"think of .{3,30} as|similar to how)\b", text
    ))
    scores["metaphor"] = metaphor
    score += min(metaphor * 1.0, 4.0)

    questions = text.count("?")
    scores["questioning"] = questions
    score += min(questions * 0.5, 3.0)

    scores["lek_score"] = round(score, 2)
    return scores


def run_distill(args):
    start = time.time()

    # Load probes
    probes = load_probes(args.prompts)
    print(f"Loaded {len(probes)} probes", file=sys.stderr)

    # Load kernel (optional — Phase 0 runs without kernel)
    kernel_text = None
    if args.kernel:
        kp = Path(args.kernel)
        if kp.exists() and kp.stat().st_size > 0:
            kernel_text = kp.read_text()
            print(f"Kernel: {len(kernel_text)} chars", file=sys.stderr)
        else:
            print("No kernel — running baseline-only (Phase 0 mode)", file=sys.stderr)
    else:
        print("No kernel — running baseline-only (Phase 0 mode)", file=sys.stderr)

    # Load model
    print(f"Loading model: {args.model}", file=sys.stderr)
    model, tokenizer = mlx_lm.load(args.model)

    # Stats
    total_generated = 0
    total_kept = 0
    score_sum = 0.0

    # Open output
    out = open(args.output, "w")

    for i, probe in enumerate(probes):
        probe_kept = 0
        probe_best = -999.0

        for sample_idx in range(args.samples):
            print(
                f"  [{i+1}/{len(probes)}] {probe['id']} sample {sample_idx+1}/{args.samples}",
                file=sys.stderr, end="", flush=True
            )

            # Build prompt — with kernel as system message, or bare prompt
            if kernel_text:
                messages = [
                    {"role": "system", "content": kernel_text},
                    {"role": "user", "content": probe["prompt"]},
                ]
            else:
                messages = [{"role": "user", "content": probe["prompt"]}]

            try:
                chat_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                # Fallback: prepend kernel to user message (if kernel exists)
                if kernel_text:
                    fallback = [{"role": "user", "content": kernel_text + "\n\n" + probe["prompt"]}]
                else:
                    fallback = [{"role": "user", "content": probe["prompt"]}]
                chat_prompt = tokenizer.apply_chat_template(
                    fallback, tokenize=False, add_generation_prompt=True
                )

            sampler = make_sampler(temp=args.temperature, top_p=args.top_p)

            t0 = time.time()
            response = mlx_lm.generate(
                model, tokenizer, prompt=chat_prompt,
                max_tokens=args.max_tokens,
                sampler=sampler,
            )
            elapsed = time.time() - t0
            total_generated += 1

            # Score
            h = score_heuristic(response)
            lek_score = h["lek_score"]
            probe_best = max(probe_best, lek_score)

            if lek_score >= args.threshold:
                # Write training example (messages format for MLX fine-tuning)
                training_example = {
                    "messages": [
                        {"role": "user", "content": probe["prompt"]},
                        {"role": "assistant", "content": response},
                    ]
                }
                # Also write metadata for tracking
                line = {
                    "type": "training",
                    "training": training_example,
                    "meta": {
                        "probe_id": probe["id"],
                        "category": probe["category"],
                        "sample_idx": sample_idx,
                        "lek_score": lek_score,
                        "chars": len(response),
                        "time_s": round(elapsed, 1),
                        "model": args.model,
                        "threshold": args.threshold,
                        "temperature": args.temperature,
                        "ts": datetime.now(timezone.utc).isoformat(),
                    }
                }
                out.write(json.dumps(line) + "\n")
                out.flush()

                total_kept += 1
                probe_kept += 1
                score_sum += lek_score
                print(f" -> {lek_score:.1f} KEPT ({len(response)} chars, {elapsed:.1f}s)", file=sys.stderr)
            else:
                print(f" -> {lek_score:.1f} skip ({len(response)} chars, {elapsed:.1f}s)", file=sys.stderr)

        # Probe summary
        print(
            f"  {probe['id']}: kept {probe_kept}/{args.samples}, best={probe_best:.1f}",
            file=sys.stderr
        )

    # Final summary
    elapsed_total = time.time() - start
    summary = {
        "type": "summary",
        "model": args.model,
        "probes": len(probes),
        "samples_per_probe": args.samples,
        "total_generated": total_generated,
        "total_kept": total_kept,
        "keep_rate": round(total_kept / max(total_generated, 1) * 100, 1),
        "avg_kept_score": round(score_sum / max(total_kept, 1), 2),
        "threshold": args.threshold,
        "temperature": args.temperature,
        "duration_s": round(elapsed_total),
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    out.write(json.dumps(summary) + "\n")
    out.close()

    print(f"\n=== Self-Distillation Complete ===", file=sys.stderr)
    print(f"Generated: {total_generated}", file=sys.stderr)
    print(f"Kept:      {total_kept} ({summary['keep_rate']}%)", file=sys.stderr)
    print(f"Avg score: {summary['avg_kept_score']}", file=sys.stderr)
    print(f"Duration:  {round(elapsed_total)}s ({round(elapsed_total/60)}m)", file=sys.stderr)
    print(f"Output:    {args.output}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Self-distillation for LEK curriculum")
    parser.add_argument("--model", required=True, help="Model path or HF ID")
    parser.add_argument("--kernel", default=None, help="Path to kernel file (JSON). Omit for Phase 0.")
    parser.add_argument("--prompts", required=True, help="Path to probes JSON")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--samples", type=int, default=10, help="Samples per probe (default: 10)")
    parser.add_argument("--threshold", type=float, default=24.0, help="v2 score threshold (default: 24.0)")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens per response (default: 4096)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (default: 0.8)")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling (default: 0.95)")
    args = parser.parse_args()

    run_distill(args)


if __name__ == "__main__":
    main()
