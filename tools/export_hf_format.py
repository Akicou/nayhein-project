#!/usr/bin/env python3
"""Export custom DualModeModel checkpoint to Hugging Face format for QLoRA compatibility."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pretrain import DualModeModel


def export_dualmode_to_hf(
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    model_type: str = "nayhein_mini",
) -> None:
    """
    Export a custom DualModeModel checkpoint (*.pt files) to Hugging Face format.
    
    This enables loading with bitsandbytes 4-bit quantization via:
        AutoModelForCausalLM.from_pretrained(output_dir, load_in_4bit=True, ...)
    
    The export:
    - Converts DualModeModel to a Hugging Face-compatible CausalLM
    - Saves model weights as pytorch_model.bin
    - Creates config.json with proper architecture definition
    - Copies/creates tokenizer files
    
    Args:
        checkpoint_dir: Path to custom checkpoint (must contain model.pt, config.pt)
        output_dir: Output directory for HF-format model
        model_type: Model type identifier used in config
    """
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not (checkpoint_dir / "model.pt").exists() or not (checkpoint_dir / "config.pt").exists():
        raise FileNotFoundError(f"Invalid checkpoint: expected model.pt and config.pt in {checkpoint_dir}")
    
    config_dict = torch.load(checkpoint_dir / "config.pt", map_location="cpu")
    model_cfg = config_dict.get("target_config", config_dict)
    
    base_vocab = model_cfg.get("original_vocab_size") or model_cfg.get("base_vocab_size") or model_cfg.get("vocab_size", 16001) - 1
    hidden_size = model_cfg.get("hidden_size", 256)
    num_layers = model_cfg.get("num_layers", 6)
    num_heads = model_cfg.get("num_heads", 4)
    head_dim = model_cfg.get("head_dim", 64)
    max_seq_len = model_cfg.get("max_seq_len", 4096)
    mlp_ratio = model_cfg.get("mlp_ratio", 4.0)
    mtp_enabled = model_cfg.get("mtp_enabled", False)
    mtp_num_heads = model_cfg.get("mtp_num_heads", 0)
    
    state_dict = torch.load(checkpoint_dir / "model.pt", map_location="cpu")
    
    print(f"Exporting {num_layers}L x {hidden_size}H model ({num_heads} heads, {head_dim} head_dim) to HF format...")
    
    hf_config = {
        "model_type": model_type,
        "architectures": ["NayheinForCausalLM"],
        "vocab_size": base_vocab + 1,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "head_dim": head_dim if head_dim > 0 else max(1, hidden_size // max(1, num_heads)),
        "intermediate_size": int(hidden_size * mlp_ratio),
        "max_position_embeddings": max_seq_len,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "rope_theta": 10000.0,
        "rope_scaling": None,
        "attention_bias": False,
        "mlp_bias": False,
        "pad_token_id": 0,
        "bos_token_id": getattr(config_dict, "bos_token_id", 1),
        "eos_token_id": getattr(config_dict, "eos_token_id", 2),
        "_name_or_path": str(output_dir.absolute()),
    }
    
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(hf_config, f, indent=2)
    
    print("Created config.json")
    
    hf_state_dict = {}
    
    if "embeddings.weight" in state_dict:
        hf_state_dict["model.embed_tokens.weight"] = state_dict["embeddings.weight"]
    if "layers.0.attention.q_proj.weight" in state_dict or "layers.0.attention.q_proj.weight" not in state_dict:
        for key, value in state_dict.items():
            if key.startswith("layers."):
                parts = key.split(".")
                if len(parts) >= 5 and parts[3] in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
                    layer_idx = parts[1]
                    module_type = "self_attn" if parts[2] == "attention" else "mlp"
                    param_name = parts[3]
                    new_key = f"model.layers.{layer_idx}.{module_type}.{param_name}.weight"
                    hf_state_dict[new_key] = value
                elif len(parts) >= 4 and parts[2] == "input_layernorm":
                    layer_idx = parts[1]
                    new_key = f"model.layers.{layer_idx}.input_layernorm.weight"
                    hf_state_dict[new_key] = value
                elif len(parts) >= 4 and parts[2] == "post_attention_layernorm":
                    layer_idx = parts[1]
                    new_key = f"model.layers.{layer_idx}.post_attention_layernorm.weight"
                    hf_state_dict[new_key] = value
                elif len(parts) >= 4 and parts[2] == "norm" and parts[1].isdigit():
                    layer_idx = parts[1]
                    new_key = f"model.layers.{layer_idx}.post_attention_layernorm.weight"
                    hf_state_dict[new_key] = value
            elif key.startswith("norm."):
                new_key = key.replace("norm.", "model.norm.")
                hf_state_dict[new_key] = value
            elif key == "lm_head.weight":
                hf_state_dict["lm_head.weight"] = value
    
    torch.save(hf_state_dict, output_dir / "pytorch_model.bin")
    print(f"Saved {len(hf_state_dict)} tensors to pytorch_model.bin")
    
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"]
    for fname in tokenizer_files:
        src = checkpoint_dir / fname
        if src.exists() and not (output_dir / fname).exists():
            (output_dir / fname).write_bytes(src.read_bytes())
    
    print(f"Export complete: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DualModeModel checkpoint to HF format")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input checkpoint directory")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output HF-format directory")
    parser.add_argument("--model-type", type=str, default="nayhein_mini", help="Model type identifier")
    
    args = parser.parse_args()
    export_dualmode_to_hf(args.input, args.output, args.model_type)
    print("Export finished successfully!")


if __name__ == "__main__":
    main()
