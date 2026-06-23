#!/usr/bin/env python3
"""
Reproduce Published LEM Benchmarks

This script reproduces the key benchmarks published in the LEM analysis.
It runs A/B tests on the top models and generates comparison reports.

Usage:
    python3 scripts/reproduce_benchmarks.py [--models MODEL1,MODEL2] [--quick]

Examples:
    # Reproduce all benchmarks (takes 2-4 hours)
    python3 scripts/reproduce_benchmarks.py
    
    # Quick test with just 1B and 4B models (30-60 min)
    python3 scripts/reproduce_benchmarks.py --models gemma-3-1b-it,gemma-3-4b-it --quick
    
    # Test specific models
    python3 scripts/reproduce_benchmarks.py --models gemma-3-12b-it
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.absolute()


@dataclass
class ModelConfig:
    """Configuration for a model to benchmark."""
    name: str
    hf_path: str
    params: str
    family: str
    expected_baseline: float
    expected_kernel_boost: float


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    model: str
    condition: str  # baseline, json, txt
    v2_score: float
    probe_count: int
    timestamp: str
    duration_seconds: float
    error: Optional[str] = None


@dataclass
class ReproductionReport:
    """Complete reproduction report."""
    timestamp: str
    models_tested: List[str]
    results: List[BenchmarkResult]
    summary: Dict = field(default_factory=dict)


# Model configurations for reproduction
MODEL_CONFIGS = {
    # Gemma3 family
    "gemma-3-1b-it": ModelConfig(
        name="Gemma3-1B",
        hf_path="google/gemma-3-1b-it",
        params="1B",
        family="Gemma",
        expected_baseline=17.45,
        expected_kernel_boost=1.55,
    ),
    "gemma-3-4b-it": ModelConfig(
        name="Gemma3-4B",
        hf_path="google/gemma-3-4b-it",
        params="4B",
        family="Gemma",
        expected_baseline=20.66,
        expected_kernel_boost=0.99,
    ),
    "gemma-3-12b-it": ModelConfig(
        name="Gemma3-12B",
        hf_path="google/gemma-3-12b-it",
        params="12B",
        family="Gemma",
        expected_baseline=19.73,
        expected_kernel_boost=5.47,
    ),
    "gemma-3-27b-it": ModelConfig(
        name="Gemma3-27B",
        hf_path="google/gemma-3-27b-it",
        params="27B",
        family="Gemma",
        expected_baseline=20.46,
        expected_kernel_boost=2.79,
    ),
    
    # Mistral family
    "mistral-7b-v0.3": ModelConfig(
        name="Mistral-7B-v0.3",
        hf_path="mistralai/Mistral-7B-v0.3",
        params="7B",
        family="Mistral",
        expected_baseline=14.58,
        expected_kernel_boost=1.78,
    ),
    
    # Qwen family
    "qwen-3-8b": ModelConfig(
        name="Qwen3-8B",
        hf_path="Qwen/Qwen3-8B",
        params="8B",
        family="Qwen",
        expected_baseline=17.35,
        expected_kernel_boost=2.0,
    ),
    
    # Llama family
    "llama-3.1-8b": ModelConfig(
        name="Llama-3.1-8B",
        hf_path="meta-llama/Llama-3.1-8B",
        params="8B",
        family="Llama",
        expected_baseline=11.28,
        expected_kernel_boost=0.88,
    ),
}

# LEK-tuned models (for comparison)
LEK_MODEL_CONFIGS = {
    "lek-gemma3-1b": ModelConfig(
        name="LEK-Gemma3-1B",
        hf_path="lthn/LEK-Gemma3-1B-layered",
        params="1B",
        family="Gemma",
        expected_baseline=22.02,
        expected_kernel_boost=0.0,  # Should degrade with kernel
    ),
    "lek-mistral-7b": ModelConfig(
        name="LEK-Mistral-7B",
        hf_path="lthn/LEK-Mistral-7B-v0.3",
        params="7B",
        family="Mistral",
        expected_baseline=21.69,
        expected_kernel_boost=0.0,
    ),
}


def run_ab_test(
    model: str,
    kernel: Optional[str] = None,
    txt_kernel: Optional[str] = None,
    prompts: str = "seeds/P01-P100.json",
    output: str = "benchmarks/reproduce.jsonl",
    max_tokens: int = 1024,
    temperature: float = 0.7,
    batch_size: int = 1,
    timeout: int = 3600,
) -> Tuple[bool, str, float]:
    """
    Run an A/B test and return (success, output_path, duration).
    """
    start_time = time.time()
    
    # Build command
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "ab_test.py"),
        f"--model={model}",
        f"--prompts={prompts}",
        f"--output={output}",
        f"--max-tokens={max_tokens}",
        f"--temperature={temperature}",
        f"--batch-size={batch_size}",
    ]
    
    if kernel:
        cmd.append(f"--kernel=json={kernel}")
    if txt_kernel:
        cmd.append(f"--kernel=txt={txt_kernel}")
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        duration = time.time() - start_time
        
        if result.returncode != 0:
            print(f"Error running A/B test:")
            print(result.stderr)
            return False, output, duration
        
        print(f"✓ Completed in {duration:.1f} seconds")
        return True, output, duration
        
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout after {timeout} seconds")
        return False, output, timeout
    except Exception as e:
        print(f"✗ Error: {e}")
        return False, output, time.time() - start_time


def extract_scores(output_path: str) -> Dict[str, float]:
    """Extract v2 scores from A/B test output."""
    scores = {}
    
    if not Path(output_path).exists():
        return scores
    
    try:
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if 'v2_score' in entry:
                        # Extract model and condition from filename or metadata
                        condition = entry.get('condition', 'baseline')
                        model = entry.get('model', 'unknown')
                        key = f"{model}_{condition}"
                        scores[key] = float(entry['v2_score'])
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Warning: Could not extract scores from {output_path}: {e}")
    
    return scores


def run_single_model_benchmark(
    model_key: str,
    config: ModelConfig,
    use_kernel: bool = False,
    use_txt_kernel: bool = False,
    quick: bool = False,
) -> Tuple[bool, BenchmarkResult]:
    """Run benchmark for a single model under specific conditions."""
    start_time = time.time()
    
    # Determine probe set
    probes = "seeds/P01-P20.json" if quick else "seeds/P01-P100.json"
    
    # Determine output file
    condition = "txt" if use_txt_kernel else ("json" if use_kernel else "baseline")
    output_file = f"benchmarks/reproduce/{model_key}_{condition}_{'p20' if quick else 'p100'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Run the test
    kernel_path = str(PROJECT_ROOT / "kernel" / "axioms.json") if use_kernel else None
    txt_kernel_path = str(PROJECT_ROOT / "kernel" / "lek-1-kernel.txt") if use_txt_kernel else None
    
    success, _, duration = run_ab_test(
        model=config.hf_path,
        kernel=kernel_path,
        txt_kernel=txt_kernel_path,
        prompts=probes,
        output=output_file,
        max_tokens=512 if quick else 1024,
        batch_size=1,
    )
    
    # Extract score
    scores = extract_scores(output_file)
    v2_score = 0.0
    for key, value in scores.items():
        if model_key in key and condition in key:
            v2_score = value
            break
    
    # Count probes
    probe_count = 20 if quick else 100
    
    result = BenchmarkResult(
        model=config.name,
        condition=condition,
        v2_score=v2_score,
        probe_count=probe_count,
        timestamp=datetime.now().isoformat(),
        duration_seconds=duration,
        error=None if success else "Test failed",
    )
    
    return success, result


def run_full_benchmark(
    model_key: str,
    config: ModelConfig,
    quick: bool = False,
) -> List[BenchmarkResult]:
    """Run full benchmark (baseline + json kernel + txt kernel) for a model."""
    print(f"\n{'='*60}")
    print(f"Benchmarking {config.name} ({config.params})")
    print(f"{'='*60}")
    
    results = []
    
    # Baseline
    print(f"\n→ Running baseline test...")
    success, result = run_single_model_benchmark(
        model_key, config, use_kernel=False, use_txt_kernel=False, quick=quick
    )
    results.append(result)
    print(f"  Baseline v2 score: {result.v2_score:.2f}")
    
    # JSON kernel
    print(f"\n→ Running with JSON kernel...")
    success, result = run_single_model_benchmark(
        model_key, config, use_kernel=True, use_txt_kernel=False, quick=quick
    )
    results.append(result)
    print(f"  JSON kernel v2 score: {result.v2_score:.2f}")
    
    # TXT kernel
    print(f"\n→ Running with TXT kernel...")
    success, result = run_single_model_benchmark(
        model_key, config, use_kernel=False, use_txt_kernel=True, quick=quick
    )
    results.append(result)
    print(f"  TXT kernel v2 score: {result.v2_score:.2f}")
    
    return results


def generate_report(results: List[BenchmarkResult], models_tested: List[str]) -> ReproductionReport:
    """Generate a reproduction report from benchmark results."""
    report = ReproductionReport(
        timestamp=datetime.now().isoformat(),
        models_tested=models_tested,
        results=results,
    )
    
    # Calculate summary statistics
    summary = {}
    
    # Group by model
    model_results = {}
    for result in results:
        if result.model not in model_results:
            model_results[result.model] = {}
        model_results[result.model][result.condition] = result
    
    # Calculate deltas
    for model, conditions in model_results.items():
        if 'baseline' in conditions and 'json' in conditions:
            delta = conditions['json'].v2_score - conditions['baseline'].v2_score
            summary[f"{model}_json_delta"] = delta
        if 'baseline' in conditions and 'txt' in conditions:
            delta = conditions['txt'].v2_score - conditions['baseline'].v2_score
            summary[f"{model}_txt_delta"] = delta
    
    report.summary = summary
    
    return report


def save_report(report: ReproductionReport, path: str) -> None:
    """Save report to JSON file."""
    data = {
        "timestamp": report.timestamp,
        "models_tested": report.models_tested,
        "results": [
            {
                "model": r.model,
                "condition": r.condition,
                "v2_score": r.v2_score,
                "probe_count": r.probe_count,
                "timestamp": r.timestamp,
                "duration_seconds": r.duration_seconds,
                "error": r.error,
            }
            for r in report.results
        ],
        "summary": report.summary,
    }
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✓ Report saved to {path}")


def print_report(report: ReproductionReport) -> None:
    """Print a formatted reproduction report."""
    print(f"\n{'='*60}")
    print("REPRODUCTION REPORT")
    print(f"{'='*60}")
    print(f"Generated: {report.timestamp}")
    print(f"Models tested: {', '.join(report.models_tested)}")
    
    # Group results by model
    model_results = {}
    for result in report.results:
        if result.model not in model_results:
            model_results[result.model] = {}
        model_results[result.model][result.condition] = result
    
    print(f"\n{'Model':<20} {'Condition':<10} {'v2 Score':<10} {'Probes':<8} {'Time (s)':<10}")
    print("-" * 70)
    
    for model, conditions in sorted(model_results.items()):
        for condition, result in sorted(conditions.items()):
            error_str = " (ERROR)" if result.error else ""
            print(f"{model:<20} {condition:<10} {result.v2_score:<10.2f} {result.probe_count:<8} {result.duration_seconds:<10.1f}{error_str}")
    
    print(f"\n{'Summary':<20} {'Value':<10}")
    print("-" * 30)
    for key, value in sorted(report.summary.items()):
        print(f"{key:<20} {value:<10.2f}")
    
    print(f"\n{'='*60}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Reproduce published LEM benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--models',
        default='gemma-3-1b-it,gemma-3-4b-it',
        help='Comma-separated list of models to test (default: gemma-3-1b-it,gemma-3-4b-it)',
    )
    parser.add_argument(
        '--include-lek',
        action='store_true',
        help='Also test LEK-tuned models',
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Use P20 (20 probes) instead of P100 for faster testing',
    )
    parser.add_argument(
        '--output-dir',
        default='benchmarks/reproduce',
        help='Directory to save results',
    )
    parser.add_argument(
        '--report',
        default='reproduction_report.json',
        help='Report filename',
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=1024,
        help='Maximum tokens per response',
    )
    
    args = parser.parse_args()
    
    # Parse models
    model_keys = [m.strip() for m in args.models.split(',')]
    
    # Validate models
    valid_models = list(MODEL_CONFIGS.keys())
    if args.include_lek:
        valid_models.extend(LEK_MODEL_CONFIGS.keys())
    
    invalid = [m for m in model_keys if m not in valid_models]
    if invalid:
        print(f"Warning: Unknown models: {', '.join(invalid)}")
        print(f"Valid models: {', '.join(valid_models)}")
        return 1
    
    # Add LEK models if requested
    if args.include_lek:
        model_keys.extend(["lek-gemma3-1b", "lek-mistral-7b"])
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print_header("LEM Benchmark Reproduction")
    print(f"Models to test: {', '.join(model_keys)}")
    print(f"Probe set: {'P20 (quick)' if args.quick else 'P100 (full)'}")
    print(f"Output directory: {output_dir}")
    
    # Estimate time
    estimated_time = len(model_keys) * 3 * (60 if args.quick else 600)  # 3 conditions per model
    print(f"Estimated time: {estimated_time/60:.0f} minutes")
    
    # Ask for confirmation
    if not args.quick and len(model_keys) > 2:
        response = input("\nThis will take several hours. Continue? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return 0
    
    # Run benchmarks
    all_results = []
    models_tested = []
    
    for model_key in model_keys:
        if model_key in MODEL_CONFIGS:
            config = MODEL_CONFIGS[model_key]
        elif model_key in LEK_MODEL_CONFIGS:
            config = LEK_MODEL_CONFIGS[model_key]
        else:
            continue
        
        models_tested.append(config.name)
        results = run_full_benchmark(model_key, config, quick=args.quick)
        all_results.extend(results)
    
    # Generate report
    report = generate_report(all_results, models_tested)
    
    # Save report
    report_path = output_dir / args.report
    save_report(report, str(report_path))
    
    # Print report
    print_report(report)
    
    # Print comparison with expected results
    print(f"\n{'='*60}")
    print("COMPARISON WITH PUBLISHED RESULTS")
    print(f"{'='*60}")
    
    for model_key, config in MODEL_CONFIGS.items():
        if model_key not in model_keys:
            continue
        
        model_results = [r for r in all_results if r.model == config.name]
        if not model_results:
            continue
        
        baseline = next((r.v2_score for r in model_results if r.condition == 'baseline'), 0)
        json_kernel = next((r.v2_score for r in model_results if r.condition == 'json'), 0)
        
        expected_baseline = config.expected_baseline
        expected_boost = config.expected_kernel_boost
        
        actual_boost = json_kernel - baseline if json_kernel and baseline else 0
        
        print(f"\n{config.name} ({config.params}):")
        print(f"  Published baseline: {expected_baseline:.2f}")
        print(f"  Your baseline:      {baseline:.2f} {'✓' if abs(baseline - expected_baseline) < 2 else '⚠'}")
        print(f"  Published boost:   +{expected_boost:.2f}")
        print(f"  Your boost:        +{actual_boost:.2f} {'✓' if abs(actual_boost - expected_boost) < 1 else '⚠'}")
    
    print(f"\n{'='*60}")
    print("Reproduction complete!")
    print(f"Full report: {report_path}")
    print(f"{'='*60}")
    
    return 0


def print_header(text: str):
    print(f"\n{'='*60}")
    print(text)
    print(f"{'='*60}\n")


if __name__ == '__main__':
    sys.exit(main())
