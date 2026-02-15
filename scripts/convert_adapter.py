"""MLX → PEFT adapter conversion.

MLX LoRA adapters use different tensor naming and layout than HuggingFace PEFT.
This module handles the three required transformations:

1. Key renaming: model.layers.N.module.lora_a → base_model.model.model.layers.N.module.lora_A.default.weight
2. Transpose: MLX stores (in, rank) / (rank, out), PEFT expects (rank, in) / (out, rank)
3. Config generation: adapter_config.json with lora_alpha = scale × rank
"""

import json
import os
import re
import shutil
import tempfile

import torch
from safetensors.torch import load_file, save_file


def rename_key(mlx_key: str) -> str:
    """Rename an MLX tensor key to PEFT format.

    model.layers.12.self_attn.q_proj.lora_a
    → base_model.model.model.layers.12.self_attn.q_proj.lora_A.default.weight
    """
    key = mlx_key
    # lora_a → lora_A.default.weight, lora_b → lora_B.default.weight
    key = re.sub(r'\.lora_a$', '.lora_A.default.weight', key)
    key = re.sub(r'\.lora_b$', '.lora_B.default.weight', key)
    # Prepend base_model.model.
    key = "base_model.model." + key
    return key


def convert_mlx_to_peft(
    mlx_safetensors_path: str,
    mlx_config_path: str,
    output_dir: str,
    base_model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
) -> str:
    """Convert an MLX LoRA adapter to PEFT format.

    Args:
        mlx_safetensors_path: Path to MLX .safetensors file
        mlx_config_path: Path to MLX adapter_config.json
        output_dir: Directory to write PEFT adapter files
        base_model_name: HuggingFace model ID for config

    Returns:
        Path to the output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load MLX tensors
    mlx_tensors = load_file(mlx_safetensors_path)

    # Rename and transpose
    peft_tensors = {}
    for mlx_key, tensor in mlx_tensors.items():
        peft_key = rename_key(mlx_key)
        # Transpose: MLX (in, rank) → PEFT (rank, in) for both A and B
        peft_tensors[peft_key] = tensor.T.contiguous()

    # Save PEFT safetensors
    save_file(peft_tensors, os.path.join(output_dir, "adapter_model.safetensors"))

    # Read MLX config for LoRA parameters
    with open(mlx_config_path) as f:
        mlx_config = json.load(f)

    lora_params = mlx_config.get("lora_parameters", {})
    rank = lora_params.get("rank", 8)
    scale = lora_params.get("scale", 20.0)
    dropout = lora_params.get("dropout", 0.0)

    # Determine target modules from tensor keys
    modules = set()
    for k in mlx_tensors:
        m = re.match(r'model\.layers\.\d+\.(.*?)\.lora_[ab]$', k)
        if m:
            # e.g. "self_attn.q_proj" → extract just the leaf: "q_proj"
            full_path = m.group(1)
            modules.add(full_path.split(".")[-1])

    # Determine layer range
    layers = set()
    for k in mlx_tensors:
        m = re.search(r'layers\.(\d+)', k)
        if m:
            layers.add(int(m.group(1)))

    # Write PEFT adapter_config.json
    peft_config = {
        "auto_mapping": None,
        "base_model_name_or_path": base_model_name,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": sorted(layers),
        "lora_alpha": scale * rank,  # MLX scale × rank = PEFT alpha
        "lora_dropout": dropout,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": rank,
        "revision": None,
        "target_modules": sorted(modules),
        "task_type": "CAUSAL_LM",
    }

    with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
        json.dump(peft_config, f, indent=2)

    return output_dir


def convert_checkpoint(
    adapter_dir: str,
    checkpoint_file: str,
    work_dir: str,
    base_model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
) -> str:
    """Convenience: convert a single MLX checkpoint file to a PEFT adapter directory.

    Args:
        adapter_dir: Directory containing MLX adapter files and config
        checkpoint_file: Filename like "0000050_adapters.safetensors"
        work_dir: Working directory for temporary PEFT output
        base_model_name: HuggingFace model ID

    Returns:
        Path to PEFT adapter directory
    """
    safetensors_path = os.path.join(adapter_dir, checkpoint_file)
    config_path = os.path.join(adapter_dir, "adapter_config.json")

    # Use checkpoint iteration as subdirectory name
    iter_match = re.search(r'(\d+)', checkpoint_file)
    iter_name = iter_match.group(1) if iter_match else "unknown"
    output_dir = os.path.join(work_dir, f"peft_{iter_name}")

    return convert_mlx_to_peft(safetensors_path, config_path, output_dir, base_model_name)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python convert_adapter.py <mlx_safetensors> <mlx_config> [output_dir]")
        print("Example: python convert_adapter.py 0000050_adapters.safetensors adapter_config.json ./peft_out")
        sys.exit(1)

    sf_path = sys.argv[1]
    cfg_path = sys.argv[2]
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "./peft_output"

    result = convert_mlx_to_peft(sf_path, cfg_path, out_dir)
    print(f"Converted to: {result}")
    print(f"Files: {os.listdir(result)}")
