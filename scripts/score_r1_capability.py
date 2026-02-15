#!/usr/bin/env python3
"""R1 Capability Benchmark — Pure reasoning, zero politics.

Tests whether LEK/sovereignty training improved or degraded the model's
actual THINKING ability. Math, logic, reasoning — binary right/wrong.

If capability improves alongside val loss dropping, PTSD is real and
LEK training heals it. The model didn't change its politics, it got smarter.
"""

import json
import os
import re
import sys
import time

MODEL = "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit"
ADAPTER_PATH = "/Volumes/Data/lem/adapters-deepseek-r1-7b-sovereignty"

# Pure capability probes — objective answers only
PROBES = [
    # === MATH ===
    {
        "id": "math_01",
        "category": "arithmetic",
        "prompt": "What is 347 × 29? Show your work and give the final answer.",
        "answer": "10063",
        "check": lambda r: "10063" in r.replace(",", "").replace(" ", ""),
    },
    {
        "id": "math_02",
        "category": "arithmetic",
        "prompt": "A store sells apples for $1.25 each. If I buy 17 apples and pay with a $50 bill, how much change do I get?",
        "answer": "28.75",
        "check": lambda r: "28.75" in r or "$28.75" in r,
    },
    {
        "id": "math_03",
        "category": "algebra",
        "prompt": "Solve for x: 3x + 7 = 2x - 5. What is x?",
        "answer": "-12",
        "check": lambda r: bool(re.search(r'x\s*=\s*-\s*12|=\s*-12|-12', r)),
    },
    {
        "id": "math_04",
        "category": "algebra",
        "prompt": "If f(x) = 2x² - 3x + 1, what is f(4)?",
        "answer": "21",
        "check": lambda r: bool(re.search(r'\b21\b', r)),
    },
    {
        "id": "math_05",
        "category": "probability",
        "prompt": "A bag has 3 red balls, 5 blue balls, and 2 green balls. What is the probability of drawing a blue ball? Express as a fraction and decimal.",
        "answer": "1/2 or 0.5",
        "check": lambda r: "1/2" in r or "0.5" in r or "50%" in r or "5/10" in r,
    },
    {
        "id": "math_06",
        "category": "geometry",
        "prompt": "A circle has a radius of 7cm. What is its area? Use pi = 3.14159.",
        "answer": "153.94",
        "check": lambda r: bool(re.search(r'15[34]\.9|153\.9[0-9]|154\.0|49\s*[πpi]', r)),
    },
    {
        "id": "math_07",
        "category": "sequences",
        "prompt": "What is the next number in this sequence: 2, 6, 18, 54, ...?",
        "answer": "162",
        "check": lambda r: "162" in r,
    },
    {
        "id": "math_08",
        "category": "percentages",
        "prompt": "A laptop costs $800. It's on sale for 15% off. Then you have a coupon for 10% off the sale price. What is the final price?",
        "answer": "612",
        "check": lambda r: bool(re.search(r'\$?612', r)),
    },
    # === LOGIC ===
    {
        "id": "logic_01",
        "category": "deduction",
        "prompt": "All cats are animals. All animals need water. Does a cat need water? Explain your reasoning.",
        "answer": "Yes",
        "check": lambda r: bool(re.search(r'\byes\b', r.lower())),
    },
    {
        "id": "logic_02",
        "category": "deduction",
        "prompt": "If it rains, the ground gets wet. The ground is wet. Can we conclude it rained? Why or why not?",
        "answer": "No — affirming the consequent fallacy",
        "check": lambda r: bool(re.search(r'\bno\b|\bcannot\b|\bcan\'t\b|not necessarily|fallac|other reason|doesn\'t mean', r.lower())),
    },
    {
        "id": "logic_03",
        "category": "deduction",
        "prompt": "In a room of 30 people, what is the minimum number of people that must share a birth month?",
        "answer": "3",
        "check": lambda r: bool(re.search(r'\b3\b|three', r.lower())) and not re.search(r'\b30\b', r[:50]),
    },
    {
        "id": "logic_04",
        "category": "puzzles",
        "prompt": "A farmer needs to cross a river with a fox, a chicken, and a bag of grain. The boat only holds the farmer and one item. If left alone, the fox eats the chicken, and the chicken eats the grain. What is the first thing the farmer should take across?",
        "answer": "The chicken",
        "check": lambda r: bool(re.search(r'chicken|hen', r.lower())),
    },
    {
        "id": "logic_05",
        "category": "sets",
        "prompt": "In a class of 40 students, 25 play football, 20 play basketball, and 10 play both. How many play neither?",
        "answer": "5",
        "check": lambda r: bool(re.search(r'\b5\b|five', r.lower())),
    },
    # === REASONING ===
    {
        "id": "reason_01",
        "category": "analogy",
        "prompt": "Complete the analogy: Book is to reading as fork is to ___",
        "answer": "eating",
        "check": lambda r: bool(re.search(r'eating|food|dining', r.lower())),
    },
    {
        "id": "reason_02",
        "category": "causal",
        "prompt": "A car won't start. The battery is new. The fuel tank is full. The starter motor clicks but the engine doesn't turn. What is the most likely problem?",
        "answer": "Starter motor / solenoid",
        "check": lambda r: bool(re.search(r'starter|solenoid|connection|terminal|corros|ground|wire', r.lower())),
    },
    {
        "id": "reason_03",
        "category": "spatial",
        "prompt": "You're facing north. You turn right 90 degrees, then turn right 90 degrees again. What direction are you facing?",
        "answer": "South",
        "check": lambda r: bool(re.search(r'\bsouth\b', r.lower())),
    },
    {
        "id": "reason_04",
        "category": "temporal",
        "prompt": "Event A happened in 1995. Event B happened 12 years before Event A. Event C happened 8 years after Event B. In what year did Event C happen?",
        "answer": "1991",
        "check": lambda r: "1991" in r,
    },
    {
        "id": "reason_05",
        "category": "pattern",
        "prompt": "If APPLE = 50 (A=1, P=16, P=16, L=12, E=5), what does CAT equal using the same system?",
        "answer": "24",
        "check": lambda r: bool(re.search(r'\b24\b', r)),
    },
    # === CODE REASONING ===
    {
        "id": "code_01",
        "category": "code",
        "prompt": "What does this Python code print?\nx = [1, 2, 3, 4, 5]\nprint(x[1:3])",
        "answer": "[2, 3]",
        "check": lambda r: "[2, 3]" in r or "[2,3]" in r,
    },
    {
        "id": "code_02",
        "category": "code",
        "prompt": "What is the output?\ndef f(n):\n    if n <= 1: return n\n    return f(n-1) + f(n-2)\nprint(f(6))",
        "answer": "8",
        "check": lambda r: bool(re.search(r'\b8\b', r)),
    },
    {
        "id": "code_03",
        "category": "code",
        "prompt": "This code has a bug. What is it?\ndef average(numbers):\n    total = 0\n    for n in numbers:\n        total += n\n    return total / len(numbers)\nprint(average([]))",
        "answer": "Division by zero",
        "check": lambda r: bool(re.search(r'divis.*zero|zero.*divis|empty|len.*0|ZeroDivision', r, re.I)),
    },
    # === WORD PROBLEMS ===
    {
        "id": "word_01",
        "category": "word",
        "prompt": "A train travels at 60 km/h. Another train travels at 80 km/h in the same direction from the same station, leaving 1 hour later. How long after the second train departs will it catch the first?",
        "answer": "3 hours",
        "check": lambda r: bool(re.search(r'\b3\b.*hour|three.*hour', r.lower())),
    },
    {
        "id": "word_02",
        "category": "word",
        "prompt": "I have twice as many sisters as brothers. My sister has as many brothers as sisters. How many children are in my family? (I am male.)",
        "answer": "7",
        "check": lambda r: bool(re.search(r'\b7\b|seven', r.lower())),
    },
]


def evaluate_checkpoint(adapter_path, adapter_file, label):
    """Run all probes through a checkpoint, return accuracy."""
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler

    if adapter_file:
        import shutil
        src = os.path.join(adapter_path, adapter_file)
        dst = os.path.join(adapter_path, 'adapters.safetensors')
        if os.path.exists(src):
            shutil.copy2(src, dst)

    model, tokenizer = load(MODEL, adapter_path=adapter_path)
    sampler = make_sampler(temp=0.1)  # Low temp for deterministic reasoning

    results = {"label": label, "probes": {}, "by_category": {}}
    correct = 0
    total = 0

    for probe in PROBES:
        print(f"  [{probe['id']}]", end=" ", flush=True)

        messages = [{'role': 'user', 'content': probe['prompt']}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        response = generate(model, tokenizer, prompt=text, max_tokens=500, sampler=sampler)

        # Strip think blocks
        clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        if not clean:
            clean = response[:500]

        passed = probe["check"](clean)
        total += 1
        if passed:
            correct += 1

        cat = probe["category"]
        results["by_category"].setdefault(cat, {"correct": 0, "total": 0})
        results["by_category"][cat]["total"] += 1
        if passed:
            results["by_category"][cat]["correct"] += 1

        results["probes"][probe["id"]] = {
            "passed": passed,
            "response": clean[:300],
            "expected": probe["answer"],
        }

        status = "PASS" if passed else "FAIL"
        print(f"{status} (expected: {probe['answer']})")

    results["accuracy"] = round(correct / total * 100, 1) if total > 0 else 0
    results["correct"] = correct
    results["total"] = total

    return results


def main():
    checkpoints = sorted([f for f in os.listdir(ADAPTER_PATH) if f.endswith('_adapters.safetensors')])
    print(f"Found {len(checkpoints)} checkpoints")

    # Test key checkpoints: @50 (content best), @200, @400, @800 (val best region), @1000 (content worst), @1200, @1500/@1600 (recovery)
    key_checkpoints = [
        "0000050_adapters.safetensors",
        "0000200_adapters.safetensors",
        "0000400_adapters.safetensors",
        "0000800_adapters.safetensors",
        "0001000_adapters.safetensors",
        "0001200_adapters.safetensors",
        "0001600_adapters.safetensors",
    ]

    # Filter to only existing ones
    key_checkpoints = [c for c in key_checkpoints if c in checkpoints]

    outfile = "/Volumes/Data/lem/benchmarks/deepseek-sovereignty-capability.jsonl"
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    all_results = []
    for i, ckpt in enumerate(key_checkpoints):
        iter_n = re.search(r'(\d+)', ckpt).group()
        label = f"R1-sov @{iter_n}"

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(key_checkpoints)}] {label}")
        print(f"{'='*60}")

        result = evaluate_checkpoint(ADAPTER_PATH, ckpt, label)
        all_results.append(result)

        print(f"\n  ACCURACY: {result['accuracy']}% ({result['correct']}/{result['total']})")

        # Category breakdown
        for cat, data in sorted(result["by_category"].items()):
            pct = round(data["correct"] / data["total"] * 100) if data["total"] > 0 else 0
            print(f"    {cat}: {data['correct']}/{data['total']} ({pct}%)")

        with open(outfile, 'a') as f:
            # Can't serialize lambdas, strip them
            clean_result = {k: v for k, v in result.items()}
            f.write(json.dumps(clean_result) + "\n")

    # Comparison table
    print(f"\n{'='*70}")
    print("CAPABILITY COMPARISON — DeepSeek R1 Sovereignty Checkpoints")
    print(f"{'='*70}")

    # Collect all categories
    all_cats = sorted(set(cat for r in all_results for cat in r["by_category"]))
    header = f"{'Checkpoint':<20} {'TOTAL':>6}" + "".join(f" {c[:8]:>8}" for c in all_cats)
    print(header)
    print("-" * len(header))

    for r in all_results:
        row = f"{r['label']:<20} {r['accuracy']:>5}%"
        for cat in all_cats:
            data = r["by_category"].get(cat, {"correct": 0, "total": 0})
            if data["total"] > 0:
                pct = round(data["correct"] / data["total"] * 100)
                row += f" {pct:>7}%"
            else:
                row += f" {'N/A':>7}"
        print(row)

    # Val loss context
    val_losses = {
        "0000050": 2.495, "0000200": 2.149, "0000400": 1.992,
        "0000800": 1.791, "0001000": 1.813, "0001200": 1.692, "0001600": 1.661
    }
    print(f"\n{'Checkpoint':<20} {'Accuracy':>8} {'Val Loss':>10} {'Content':>10}")
    print("-" * 50)
    content_avgs = {
        "0000050": 7.5, "0000200": 4.9, "0000400": 4.5,
        "0000800": 4.7, "0001000": 3.1, "0001200": 4.4, "0001600": 5.8
    }
    for r in all_results:
        iter_n = re.search(r'(\d+)', r['label']).group().zfill(7)
        vl = val_losses.get(iter_n, "?")
        ca = content_avgs.get(iter_n, "?")
        print(f"{r['label']:<20} {r['accuracy']:>7}% {vl:>10} {ca:>10}")


if __name__ == "__main__":
    main()
