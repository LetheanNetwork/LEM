#!/usr/bin/env python3
"""
LEM Setup Script

This script verifies and sets up your LEM development environment.
Run with: python3 setup.py [check|install|all]
"""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).parent.absolute()


def print_header(text: str):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def print_section(text: str):
    print(f"\n{text}")
    print("-" * 40)


def check_python() -> Tuple[bool, str, str]:
    """Check Python installation."""
    try:
        result = subprocess.run(
            [sys.executable, "--version"],
            capture_output=True, text=True, check=True
        )
        version = result.stdout.strip()
        # Extract version number
        version_num = version.split()[-1]
        major, minor = version_num.split('.')[:2]
        
        if int(major) >= 3 and int(minor) >= 9:
            return True, version, f"Python {major}.{minor} is supported"
        else:
            return False, version, f"Python 3.9+ required, found {version}"
    except Exception as e:
        return False, "", f"Python not found: {e}"


def check_package(package: str) -> Tuple[bool, str, str]:
    """Check if a Python package is installed."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", f"import {package}; print({package}.__version__)"],
            capture_output=True, text=True, check=True, timeout=10
        )
        version = result.stdout.strip()
        return True, version, f"{package} {version} is installed"
    except subprocess.TimeoutExpired:
        return False, "", f"Timeout checking {package}"
    except Exception as e:
        return False, "", f"{package} not installed: {e}"


def check_go() -> Tuple[bool, str, str]:
    """Check Go installation."""
    try:
        result = subprocess.run(
            ["go", "version"],
            capture_output=True, text=True, check=True
        )
        version = result.stdout.strip()
        return True, version, f"Go {version} is installed"
    except Exception as e:
        return False, "", f"Go not found: {e}"


def check_docker() -> Tuple[bool, str, str]:
    """Check Docker installation."""
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True, text=True, check=True
        )
        version = result.stdout.strip()
        return True, version, f"Docker {version} is installed"
    except Exception as e:
        return False, "", f"Docker not found: {e}"


def check_hardware() -> Dict[str, str]:
    """Check system hardware."""
    info = {}
    
    # Platform
    info["Platform"] = platform.platform()
    info["System"] = platform.system()
    info["Release"] = platform.release()
    info["Machine"] = platform.machine()
    info["Processor"] = platform.processor()
    
    # CPU info
    try:
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, check=True
            )
            info["CPU"] = result.stdout.strip()
        elif platform.system() == "Linux":
            result = subprocess.run(
                ["lscpu"], capture_output=True, text=True, check=True
            )
            info["CPU Info"] = result.stdout.strip()
    except:
        pass
    
    # Memory
    try:
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "hw.memsize"],
                capture_output=True, text=True, check=True
            )
            mem_bytes = int(result.stdout.strip())
            info["Memory"] = f"{mem_bytes / (1024**3):.1f} GB"
        elif platform.system() == "Linux":
            result = subprocess.run(
                ["free", "-h"], capture_output=True, text=True, check=True
            )
            info["Memory"] = result.stdout.strip()
    except:
        pass
    
    # GPU info
    try:
        if platform.system() == "Darwin":
            result = subprocess.run(
                ["system_profiler", "SPMetalDataType"],
                capture_output=True, text=True, check=True, timeout=5
            )
            if "GPU" in result.stdout:
                info["GPU"] = "Apple Metal (MPS)"
        elif platform.system() == "Linux":
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, check=True
            )
            if result.returncode == 0:
                info["GPU"] = "NVIDIA GPU detected"
    except:
        pass
    
    return info


def check_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, check=True
        )
        return "Apple" in result.stdout
    except:
        return False


def check_mlx_support() -> Tuple[bool, str]:
    """Check if MLX is supported on this hardware."""
    is_apple = check_apple_silicon()
    
    if not is_apple:
        return False, "MLX requires Apple Silicon (M1/M2/M3)"
    
    try:
        import mlx
        return True, "MLX is supported on this hardware"
    except ImportError:
        return False, "MLX not installed (but hardware supports it)"
    except Exception as e:
        return False, f"MLX error: {e}"


def check_required_packages() -> List[Tuple[str, bool, str, str]]:
    """Check all required Python packages."""
    packages = [
        "mlx",
        "mlx-lm",
        "torch",
        "transformers",
        "accelerate",
        "peft",
        "datasets",
        "pandas",
        "numpy",
    ]
    
    results = []
    for pkg in packages:
        installed, version, message = check_package(pkg)
        results.append((pkg, installed, version, message))
    
    return results


def check_lem_files() -> Dict[str, List[str]]:
    """Check for essential LEM files."""
    checks = {
        "Kernels": [],
        "Seeds": [],
        "Training Data": [],
        "Benchmarks": [],
        "Scripts": [],
    }
    
    # Kernels
    kernel_dir = PROJECT_ROOT / "kernel"
    if kernel_dir.exists():
        for f in ["axioms.json", "lek-1-kernel.txt"]:
            if (kernel_dir / f).exists():
                checks["Kernels"].append(f"✓ {f}")
            else:
                checks["Kernels"].append(f"✗ {f} missing")
    else:
        checks["Kernels"].append("✗ kernel/ directory missing")
    
    # Seeds
    seeds_dir = PROJECT_ROOT / "seeds"
    if seeds_dir.exists():
        for f in ["P01-P100.json", "P01-P100-rephrased.json"]:
            if (seeds_dir / f).exists():
                checks["Seeds"].append(f"✓ {f}")
            else:
                checks["Seeds"].append(f"✗ {f} missing")
    else:
        checks["Seeds"].append("✗ seeds/ directory missing")
    
    # Training
    training_dir = PROJECT_ROOT / "training"
    if training_dir.exists():
        for f in ["train.jsonl", "valid.jsonl", "test.jsonl"]:
            if (training_dir / f).exists():
                checks["Training Data"].append(f"✓ {f}")
            else:
                checks["Training Data"].append(f"✗ {f} missing")
    else:
        checks["Training Data"].append("✗ training/ directory missing")
    
    # Benchmarks
    benchmarks_dir = PROJECT_ROOT / "benchmarks"
    if benchmarks_dir.exists():
        ab_files = list(benchmarks_dir.glob("ab-*.jsonl"))
        if ab_files:
            checks["Benchmarks"].append(f"✓ {len(ab_files)} A/B test files")
        else:
            checks["Benchmarks"].append("✗ No A/B test files found")
    else:
        checks["Benchmarks"].append("✗ benchmarks/ directory missing")
    
    # Scripts
    scripts_dir = PROJECT_ROOT / "scripts"
    if scripts_dir.exists():
        scripts = list(scripts_dir.glob("*.py"))
        checks["Scripts"].append(f"✓ {len(scripts)} Python scripts")
    else:
        checks["Scripts"].append("✗ scripts/ directory missing")
    
    return checks


def install_packages(packages: List[str]) -> int:
    """Install missing Python packages."""
    print_section("Installing missing packages")
    
    missing = []
    for pkg in packages:
        installed, _, _ = check_package(pkg)
        if not installed:
            missing.append(pkg)
    
    if not missing:
        print("All packages are already installed!")
        return 0
    
    print(f"Installing {len(missing)} packages: {', '.join(missing)}")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade"] + missing,
            check=True
        )
        print("✓ Packages installed successfully")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install packages: {e}")
        return 1


def generate_config() -> int:
    """Generate configuration files."""
    print_section("Generating configuration")
    
    # Check if config already exists
    config_file = PROJECT_ROOT / "lem.config.json"
    if config_file.exists():
        print("✓ Configuration file already exists")
        return 0
    
    # Generate default config
    config = {
        "model_dir": "data/models",
        "adapter_dir": "data/adapters",
        "training_dir": "training",
        "seeds_dir": "seeds",
        "benchmarks_dir": "benchmarks",
        "kernel_dir": "kernel",
        "default_kernel": "kernel/axioms.json",
        "default_probes": "seeds/P01-P100.json",
        "default_model": "gemma-3-1b-it",
        "backend": "auto",
        "device": "auto",
        "max_tokens": 1024,
        "temperature": 0.7,
        "batch_size": 1,
    }
    
    try:
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Configuration saved to {config_file}")
        return 0
    except Exception as e:
        print(f"✗ Failed to save configuration: {e}")
        return 1


def verify_setup() -> int:
    """Verify the complete LEM setup."""
    print_header("LEM Setup Verification")
    
    all_good = True
    
    # 1. Python
    print_section("Python Environment")
    python_ok, python_ver, python_msg = check_python()
    print(f"  {python_msg}")
    if not python_ok:
        all_good = False
    
    # 2. Hardware
    print_section("Hardware Information")
    hardware = check_hardware()
    for key, value in hardware.items():
        print(f"  {key}: {value}")
    
    is_apple = check_apple_silicon()
    mlx_ok, mlx_msg = check_mlx_support()
    print(f"  Apple Silicon: {'Yes' if is_apple else 'No'}")
    print(f"  MLX Support: {mlx_msg}")
    
    # 3. Dependencies
    print_section("Python Packages")
    packages = check_required_packages()
    for pkg, installed, version, msg in packages:
        status = "✓" if installed else "✗"
        ver_str = f" ({version})" if version else ""
        print(f"  {status} {pkg}{ver_str}")
        if not installed:
            all_good = False
    
    # 4. Go
    print_section("Go Toolchain")
    go_ok, go_ver, go_msg = check_go()
    print(f"  {go_msg}")
    if not go_ok:
        print("  Note: Go is optional for production tooling")
    
    # 5. Docker
    print_section("Docker")
    docker_ok, docker_ver, docker_msg = check_docker()
    print(f"  {docker_msg}")
    if not docker_ok:
        print("  Note: Docker is optional for deployment")
    
    # 6. LEM Files
    print_section("LEM Files")
    file_checks = check_lem_files()
    for category, items in file_checks.items():
        print(f"  {category}:")
        for item in items:
            print(f"    {item}")
            if "✗" in item:
                all_good = False
    
    # Summary
    print_header("Summary")
    if all_good:
        print("✓ All checks passed! LEM is ready to use.")
        print("\nNext steps:")
        print("  1. Run: lem info")
        print("  2. Try: lem benchmark --model gemma-3-1b-it --kernel kernel/axioms.json --prompts seeds/P01-P20.json --output test.jsonl")
        print("  3. Read: docs/QUICKSTART.md")
        return 0
    else:
        print("✗ Some checks failed. See above for details.")
        print("\nTo fix issues:")
        print("  - Install missing packages: python3 setup.py install")
        print("  - Generate config: python3 setup.py config")
        print("  - Check documentation: docs/")
        return 1


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python3 setup.py [check|install|config|all]")
        print("\nCommands:")
        print("  check   - Verify current setup")
        print("  install - Install missing dependencies")
        print("  config  - Generate configuration files")
        print("  all     - Run all setup steps")
        return 1
    
    command = sys.argv[1].lower()
    
    if command == "check" or command == "verify":
        return verify_setup()
    elif command == "install":
        return install_packages([
            "mlx", "mlx-lm", "torch", "transformers", 
            "accelerate", "peft", "datasets", "pandas", "numpy"
        ])
    elif command == "config":
        return generate_config()
    elif command == "all":
        print_header("LEM Full Setup")
        
        # Step 1: Install packages
        if install_packages(["mlx", "mlx-lm"]) != 0:
            print("\nWarning: Some packages failed to install")
        
        # Step 2: Generate config
        if generate_config() != 0:
            print("\nWarning: Config generation failed")
        
        # Step 3: Verify
        return verify_setup()
    else:
        print(f"Unknown command: {command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
