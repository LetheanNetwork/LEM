#!/usr/bin/env python3
"""Generate probe rephrasings for curriculum augmentation.

Takes the P01-P100 probe set and generates 3 variants per probe:
1. Different perspective (e.g., first person → third person, local → global)
2. Different urgency (casual reflection → urgent crisis)
3. Different cultural context (Western → non-Western framing)

These rephrasings are used in Phase 1 self-distillation to produce
more diverse training data from the same ethical scenarios.

Usage:
  python3 rephrase_probes.py \
    --input /Volumes/Data/lem/seeds/P01-P100.json \
    --output /Volumes/Data/lem/seeds/P01-P100-rephrased.json
"""

import argparse
import json


# Rephrase templates — transform the framing without changing the core scenario
REPHRASE_TEMPLATES = {
    "perspective": [
        "You are a journalist investigating the following situation. Write your findings as a long-form article:\n\n{prompt}",
        "A community organiser has asked for your advice on this situation. Write a detailed response that considers all stakeholders:\n\n{prompt}",
        "You are writing a university lecture on this topic. Present the scenario as a case study with analysis:\n\n{prompt}",
    ],
    "urgency": [
        "This is happening right now, today, in your community. A decision needs to be made by tomorrow. How would you approach it?\n\n{prompt}",
        "Looking back from fifty years in the future, a historian is writing about this period. What would they say about how we handled this?\n\n{prompt}",
        "A child asks you to explain this situation to them. They want to understand why it's complicated. How do you explain it honestly without oversimplifying?\n\n{prompt}",
    ],
    "cultural": [
        "Consider this scenario from the perspective of a small island nation in the Pacific. How do the dynamics change when resources are limited and community ties are strong?\n\n{prompt}",
        "This scenario is playing out simultaneously in Lagos, Mumbai, and São Paulo. What does each city's version look like, and what do they share in common?\n\n{prompt}",
        "An Indigenous elder and a Silicon Valley entrepreneur are both asked about this situation. Write both responses, then write what they might say to each other:\n\n{prompt}",
    ],
}


def rephrase(args):
    with open(args.input) as f:
        probes = json.load(f)

    output = list(probes)  # Start with originals

    for probe in probes:
        probe_id = probe["id"]
        original_prompt = probe["prompt"]
        category = probe.get("category", probe.get("domain", "uncategorised"))

        # Generate one variant per rephrase category
        for rcat, templates in REPHRASE_TEMPLATES.items():
            # Cycle through templates based on probe index
            idx = int(probe_id.split("_")[0].replace("P", "")) if probe_id.startswith("P") else 0
            template = templates[idx % len(templates)]

            variant = {
                "id": f"{probe_id}_R_{rcat.upper()[:4]}",
                "category": category,
                "domain": probe.get("domain", category),
                "prompt": template.format(prompt=original_prompt),
                "source_probe": probe_id,
                "rephrase_type": rcat,
            }
            output.append(variant)

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    originals = len(probes)
    variants = len(output) - originals
    print(f"Originals: {originals}")
    print(f"Variants:  {variants} ({variants // originals} per probe)")
    print(f"Total:     {len(output)}")
    print(f"Output:    {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Generate probe rephrasings")
    parser.add_argument("--input", required=True, help="Input probes JSON")
    parser.add_argument("--output", required=True, help="Output probes JSON (originals + variants)")
    args = parser.parse_args()
    rephrase(args)


if __name__ == "__main__":
    main()
