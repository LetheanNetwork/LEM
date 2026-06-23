#!/usr/bin/env python3
"""
Train Mistral Models with LEK

This script provides a streamlined workflow for training Mistral models with the LEK kernel.
It handles the complete pipeline: data preparation, training, evaluation, and fusion.

Usage:
    python3 scripts/train_mistral_lek.py [--model MODEL] [--phase PHASE] [--quick]

Examples:
    # Full training pipeline for Mistral-7B
    python3 scripts/train_mistral_lek.py --model mistral-7b-v0.3
    
    # Quick test with just Phase 0
    python3 scripts/train_mistral_lek.py --model mistral-7b-v0.3 --phase 0 --quick
    
    # Resume training from a specific phase
    python3 scripts/train_mistral_lek.py --model mistral-7b-v0.3 --phase 2 --resume
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
class TrainingConfig:
    """Configuration for a training run."""
    model_name: str
    hf_path: str
    params: str
    phases: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    iters_per_phase: Dict[int, int] = field(default_factory=lambda: {
        0: 200,  # P0: Ethics 0
        1: 200,  # P1: Zen
        2: 200,  # P2: Ethics 1
        3: 200,  # P3: Ethics 2+
        4: 200,  # P4: Tension
        5: 200,  # P5: Creative
    })
    batch_size: int = 2
    learning_rate: float = 1e-5
    save_every: int = 50
    max_tokens: int = 1024
    temperature: float = 0.7


# Mistral model configurations
MISTRAL_CONFIGS = {
    "mistral-7b-v0.1": TrainingConfig(
        model_name="Mistral-7B-v0.1",
        hf_path="mistralai/Mistral-7B-v0.1",
        params="7B",
        phases=[0, 1, 2, 3],  # v0.1 may need fewer phases
    ),
    "mistral-7b-v0.2": TrainingConfig(
        model_name="Mistral-7B-v0.2",
        hf_path="mistralai/Mistral-7B-v0.2",
        params="7B",
        phases=[0, 1, 2, 3],
    ),
    "mistral-7b-v0.3": TrainingConfig(
        model_name="Mistral-7B-v0.3",
        hf_path="mistralai/Mistral-7B-v0.3",
        params="7B",
        phases=[0, 1, 2, 3, 4, 5],
    ),
    "mistral-7b-instruct-v0.3": TrainingConfig(
        model_name="Mistral-7B-Instruct-v0.3",
        hf_path="mistralai/Mistral-7B-Instruct-v0.3",
        params="7B",
        phases=[0, 1, 2, 3, 4, 5],
    ),
}

# Phase configurations
PHASE_CONFIGS = {
    0: {
        "name": "Ethics 0",
        "description": "Initial axiom absorption via sandwich",
        "format": "sandwich",
        "data": "training/lem/ethics/core.jsonl",
        "probes": "seeds/P01-P100.json",
        "target_score": 18.0,
    },
    1: {
        "name": "Zen",
        "description": "Philosophical substrate without LEK",
        "format": "freeflow",
        "data": "training/lem/zen/lessons/0-allen.jsonl",
        "probes": "seeds/P01-P100.json",
        "target_score": 18.0,
    },
    2: {
        "name": "Ethics 1",
        "description": "Deeper alignment via sandwich",
        "format": "sandwich",
        "data": "training/lem/ethics/expanded.jsonl",
        "probes": "seeds/P01-P100.json",
        "target_score": 19.0,
    },
    3: {
        "name": "Ethics 2+",
        "description": "Freeflow validation",
        "format": "freeflow",
        "data": "training/lem/ethics/adversarial.jsonl",
        "probes": "seeds/P01-P100.json",
        "target_score": 19.0,
    },
    4: {
        "name": "Tension",
        "description": "Geopolitical multi-perspective scenarios",
        "format": "freeflow",
        "data": "training/lem/tension/scenarios.jsonl",
        "probes": "seeds/P01-P100.json",
        "target_score": 20.0,
    },
    5: {
        "name": "Creative",
        "description": "Voice and style development",
        "format": "freeflow",
        "data": "training/lem/creative/probes.jsonl",
        "probes": "seeds/P01-P100.json",
        "target_score": 20.0,
    },
}


def check_dependencies() -> Tuple[bool, List[str]]:
    """Check for required dependencies."""
    missing = []
    
    # Check Python packages
    packages = ["torch", "transformers", "accelerate", "peft", "datasets"]
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(f"Python package: {pkg}")
    
    # Check for GPU or MPS
    try:
        import torch
        if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
            missing.append("GPU or MPS (Apple Silicon) for acceleration")
    except:
        missing.append("PyTorch")
    
    return len(missing) == 0, missing


def setup_directories(config: TrainingConfig) -> Dict[str, Path]:
    """Set up directory structure for training."""
    base_dir = PROJECT_ROOT / "data" / "lem" / "mistral" / config.model_name.lower().replace("-", "_")
    
    dirs = {
        "base": base_dir / "base",
        "adapters": base_dir / "adapters",
        "checkpoints": base_dir / "checkpoints",
        "fused": base_dir / "fused",
        "logs": base_dir / "logs",
        "results": base_dir / "results",
    }
    
    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def download_model(config: TrainingConfig, dirs: Dict[str, Path]) -> Tuple[bool, str]:
    """Download base model using transformers."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Downloading {config.hf_path}...")
    
    try:
        # Download and save model
        model_path = dirs["base"] / "model"
        
        tokenizer = AutoTokenizer.from_pretrained(config.hf_path)
        model = AutoModelForCausalLM.from_pretrained(
            config.hf_path,
            torch_dtype="auto",
            device_map="auto",
        )
        
        # Save model
        model.save_pretrained(str(model_path))
        tokenizer.save_pretrained(str(model_path))
        
        print(f"✓ Model saved to {model_path}")
        return True, str(model_path)
        
    except Exception as e:
        print(f"✗ Failed to download model: {e}")
        return False, ""


def convert_to_mlx(model_path: str, output_path: str) -> bool:
    """Convert model to MLX format for Apple Silicon."""
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "mlx_lm.convert",
                f"--hf-path={model_path}",
                f"--mlx-path={output_path}",
                "-q",
            ],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        
        if result.returncode == 0:
            print(f"✓ Model converted to MLX format at {output_path}")
            return True
        else:
            print(f"✗ Conversion failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Conversion error: {e}")
        return False


def prepare_training_data(phase: int, dirs: Dict[str, Path]) -> Tuple[bool, str]:
    """Prepare training data for a specific phase."""
    phase_config = PHASE_CONFIGS.get(phase)
    if not phase_config:
        print(f"✗ Unknown phase: {phase}")
        return False, ""
    
    # Check if data file exists
    data_path = PROJECT_ROOT / phase_config["data"]
    if not data_path.exists():
        print(f"✗ Training data not found: {data_path}")
        # Try to generate it
        if phase == 0:
            print("  Generating Phase 0 data...")
            if generate_phase0_data(dirs):
                return True, str(data_path)
        return False, ""
    
    return True, str(data_path)


def generate_phase0_data(dirs: Dict[str, Path]) -> bool:
    """Generate Phase 0 (Ethics 0) training data."""
    try:
        # Use the self_distill script to generate training data
        result = subprocess.run(
            [
                sys.executable, str(PROJECT_ROOT / "scripts" / "self_distill.py"),
                f"--model={dirs['base']}/model",
                f"--kernel={PROJECT_ROOT}/kernel/axioms.json",
                f"--prompts={PROJECT_ROOT}/seeds/P01-P100.json",
                f"--output={PROJECT_ROOT}/training/lem/ethics/core.jsonl",
                "--samples=1",
                "--threshold=15.0",
                "--max-tokens=2048",
                "--temperature=0.7",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        
        if result.returncode == 0:
            print("✓ Phase 0 data generated")
            return True
        else:
            print(f"✗ Failed to generate Phase 0 data: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Error generating Phase 0 data: {e}")
        return False


def train_phase(
    config: TrainingConfig,
    phase: int,
    model_path: str,
    dirs: Dict[str, Path],
    quick: bool = False,
) -> Tuple[bool, str]:
    """Train a single phase."""
    phase_config = PHASE_CONFIGS.get(phase)
    if not phase_config:
        return False, ""
    
    print(f"\n{'='*60}")
    print(f"Phase {phase}: {phase_config['name']}")
    print(f"{'='*60}")
    print(f"Description: {phase_config['description']}")
    print(f"Format: {phase_config['format']}")
    print(f"Target score: {phase_config['target_score']}")
    
    # Prepare data
    data_ok, data_path = prepare_training_data(phase, dirs)
    if not data_ok:
        print(f"✗ Failed to prepare training data for Phase {phase}")
        return False, ""
    
    # Determine iterations
    iters = config.iters_per_phase.get(phase, 200)
    if quick:
        iters = min(iters, 50)
    
    # Build training command
    adapter_path = dirs["adapters"] / f"phase{phase}"
    
    # Use PEFT for training
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
        )
        from datasets import load_dataset
        import torch
        
        print(f"\nLoading model from {model_path}...")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Prepare for LoRA
        model = prepare_model_for_kbit_training(model)
        
        # LoRA config
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Add adapter
        model = get_peft_model(model, lora_config)
        
        # Load dataset
        print(f"Loading dataset from {data_path}...")
        dataset = load_dataset("json", data_files={"train": data_path})["train"]
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(adapter_path),
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=4,
            learning_rate=config.learning_rate,
            num_train_epochs=1,
            max_steps=iters,
            save_steps=config.save_every,
            logging_steps=10,
            save_total_limit=2,
            report_to="none",
            optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=lambda data: {
                'input_ids': torch.stack([f['input_ids'] for f in data]),
                'attention_mask': torch.stack([f['attention_mask'] for f in data]),
                'labels': torch.stack([f['labels'] for f in data]),
            },
        )
        
        print(f"Starting training for {iters} iterations...")
        trainer.train()
        
        # Save adapter
        trainer.save_model(str(adapter_path))
        print(f"✓ Adapter saved to {adapter_path}")
        
        return True, str(adapter_path)
        
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("  Install with: pip install peft transformers datasets accelerate")
        return False, ""
    except Exception as e:
        print(f"✗ Training error: {e}")
        import traceback
        traceback.print_exc()
        return False, ""


def evaluate_phase(
    config: TrainingConfig,
    phase: int,
    model_path: str,
    adapter_path: str,
    dirs: Dict[str, Path],
) -> Tuple[bool, float]:
    """Evaluate a trained phase."""
    phase_config = PHASE_CONFIGS.get(phase)
    if not phase_config:
        return False, 0.0
    
    print(f"\nEvaluating Phase {phase}...")
    
    try:
        # Load model with adapter
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import torch
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load adapter
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        # Run evaluation on probes
        probes_path = PROJECT_ROOT / phase_config["probes"]
        with open(probes_path, 'r') as f:
            probes = json.load(f)
        
        # Run a few probes for quick evaluation
        scores = []
        for i, probe in enumerate(probes[:10]):  # First 10 probes for quick eval
            inputs = tokenizer(probe, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Simple scoring (placeholder - use actual scorer)
            score = len(response.split()) / 10  # Very rough estimate
            scores.append(score)
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        print(f"  Average score: {avg_score:.2f}")
        
        return True, avg_score
        
    except Exception as e:
        print(f"✗ Evaluation error: {e}")
        return False, 0.0


def fuse_adapter(model_path: str, adapter_path: str, output_path: str) -> bool:
    """Fuse adapter into base model."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import torch
        
        print(f"Fusing adapter {adapter_path} into {model_path}...")
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load and fuse adapter
        model = PeftModel.from_pretrained(base_model, adapter_path)
        fused_model = model.merge_and_unload()
        
        # Save fused model
        fused_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        print(f"✓ Fused model saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"✗ Fusion error: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train Mistral models with LEK',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--model',
        default='mistral-7b-v0.3',
        choices=list(MISTRAL_CONFIGS.keys()),
        help='Mistral model to train',
    )
    parser.add_argument(
        '--phase',
        type=int,
        default=None,
        choices=list(range(6)),
        help='Specific phase to train (0-5). If not specified, trains all phases.',
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: fewer iterations, faster training',
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from last completed phase',
    )
    parser.add_argument(
        '--backend',
        choices=['transformers', 'mlx'],
        default='transformers',
        help='Backend to use for training',
    )
    parser.add_argument(
        '--device',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        default='auto',
        help='Device to use for training',
    )
    
    args = parser.parse_args()
    
    # Get model config
    config = MISTRAL_CONFIGS.get(args.model)
    if not config:
        print(f"Unknown model: {args.model}")
        print(f"Available models: {', '.join(MISTRAL_CONFIGS.keys())}")
        return 1
    
    # Check dependencies
    deps_ok, missing = check_dependencies()
    if not deps_ok:
        print("Missing dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with:")
        print("  pip install torch transformers accelerate peft datasets")
        return 1
    
    # Set up directories
    dirs = setup_directories(config)
    print(f"Directories set up at: {dirs['base'].parent}")
    
    # Determine phases to run
    if args.phase is not None:
        phases = [args.phase]
    elif args.resume:
        # Find last completed phase
        phases = []
        for p in config.phases:
            adapter_path = dirs["adapters"] / f"phase{p}"
            if not adapter_path.exists():
                phases = config.phases[config.phases.index(p):]
                break
        if not phases:
            phases = config.phases
    else:
        phases = config.phases
    
    print(f"Phases to train: {phases}")
    
    # Download or use existing model
    model_path = dirs["base"] / "model"
    if not model_path.exists():
        print(f"Downloading {config.hf_path}...")
        download_ok, model_path_str = download_model(config, dirs)
        if not download_ok:
            print("Failed to download model. Please download manually.")
            print(f"  Model: {config.hf_path}")
            print(f"  Save to: {model_path}")
            return 1
        model_path = Path(model_path_str)
    else:
        print(f"Using existing model at {model_path}")
    
    # Convert to MLX if requested
    if args.backend == 'mlx':
        mlx_path = dirs["base"] / "model_mlx"
        if not mlx_path.exists():
            if not convert_to_mlx(str(model_path), str(mlx_path)):
                print("MLX conversion failed. Falling back to transformers.")
                args.backend = 'transformers'
        else:
            model_path = mlx_path
    
    # Track current model path (may change after fusion)
    current_model_path = str(model_path)
    
    # Run training phases
    for phase in phases:
        print(f"\n{'#'*60}")
        print(f"# Starting Phase {phase}")
        print(f"{'#'*60}")
        
        # Train
        train_ok, adapter_path = train_phase(
            config, phase, current_model_path, dirs, quick=args.quick
        )
        
        if not train_ok:
            print(f"✗ Phase {phase} training failed")
            return 1
        
        # Evaluate
        eval_ok, score = evaluate_phase(
            config, phase, current_model_path, adapter_path, dirs
        )
        
        if not eval_ok:
            print(f"⚠ Phase {phase} evaluation failed (score: {score:.2f})")
        else:
            print(f"✓ Phase {phase} completed with score: {score:.2f}")
        
        # Check if we should fuse
        phase_config = PHASE_CONFIGS.get(phase)
        if phase_config and phase_config.get('fuse', True):
            # Fuse adapter into model for next phase
            fused_path = dirs["fused"] / f"phase{phase}_fused"
            if fuse_adapter(current_model_path, adapter_path, str(fused_path)):
                current_model_path = str(fused_path)
                print(f"  Model updated to fused version at {fused_path}")
            else:
                print(f"  ⚠ Fusion failed, continuing with base model")
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Model: {config.model_name}")
    print(f"Final model path: {current_model_path}")
    print(f"\nNext steps:")
    print(f"  1. Run full evaluation: python3 scripts/ab_test.py --model {current_model_path} --prompts seeds/P01-P100.json")
    print(f"  2. Compare with baseline: python3 scripts/compare_v1_v2.py")
    print(f"  3. Push to HuggingFace (optional): python3 scripts/push_all_models.py")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
