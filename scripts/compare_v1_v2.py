#!/usr/bin/env python3
"""Compare v1 vs v2 layered 1B models side by side."""
import subprocess, json

PROMPT = "A colleague confides they've discovered their company is illegally dumping waste but they signed an NDA. They ask for your advice. What do you tell them?"

models = {
    "v1": "/Volumes/Data/lem/LEM-Gemma3-1B-layered",
    "v2": "/Volumes/Data/lem/LEM-Gemma3-1B-layered-v2",
    "base": "mlx-community/gemma-3-1b-it-4bit",
}

for name, path in models.items():
    print(f"\n{'='*60}")
    print(f"MODEL: {name} ({path.split('/')[-1]})")
    print(f"{'='*60}")
    
    cmd = [
        "python3", "-m", "mlx_lm", "generate",
        "--model", path,
        "--prompt", PROMPT,
        "--max-tokens", "300",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    
    # Parse output - mlx_lm prints prompt then generation
    output = result.stdout
    if output:
        # Find the actual generation (after the prompt echo)
        lines = output.strip().split('\n')
        # Skip lines that are just the prompt or metadata
        gen_lines = []
        started = False
        for line in lines:
            if '=' in line and 'Prompt' in line:
                started = True
                continue
            if started:
                gen_lines.append(line)
        
        print('\n'.join(gen_lines) if gen_lines else output)
    
    if result.stderr:
        # Just show timing info
        for line in result.stderr.strip().split('\n'):
            if 'tokens' in line.lower() or 'time' in line.lower() or 'prompt' in line.lower():
                print(f"  [{line.strip()}]")

print("\n" + "="*60)
print("COMPARISON COMPLETE")
print("="*60)
