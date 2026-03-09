#!/usr/bin/env python3
"""
CLI helper for common finetuning operations.

Usage:
    # Finetune with YAML config
    python -m finetune --config config/finetune_4b_qlora.yaml
    
    # Export custom checkpoint to HF format (for QLoRA)
    python tools/cli.py export --input ./checkpoints/4b_scaled --output ./checkpoints/4b_scaled_hf
    
    # Estimate VRAM requirements
    python tools/cli.py estimate --model-path ./checkpoints/4b_scaled --use-qlora
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def cmd_export(args: argparse.Namespace) -> None:
    """Export custom DualModeModel checkpoint to HF format."""
    from finetune.custom_checkpoint import ensure_hf_export
    from tools.export_hf_format import export_dualmode_to_hf

    if args.force:
        ensure_hf_export(args.input, args.output, force=True)
    else:
        export_dualmode_to_hf(args.input, args.output, model_type=args.model_type)
    print(f"Export complete: {args.output}")


def cmd_estimate(args: argparse.Namespace) -> None:
    """Estimate VRAM requirements for a model."""
    checkpoint_dir = Path(args.model_path)
    
    param_count = None
    model_type = "unknown"
    
    if (checkpoint_dir / "config.pt").exists():
        import torch
        model_type = "custom_dualmode"
        config_dict = torch.load(checkpoint_dir / "config.pt", map_location="cpu")
        model_cfg = config_dict.get("target_config", config_dict)
        
        hidden_size = model_cfg.get("hidden_size", 256)
        num_layers = model_cfg.get("num_layers", 6)
        num_heads = model_cfg.get("num_heads", 4)
        vocab_size = model_cfg.get("vocab_size", 16001)
        mlp_ratio = model_cfg.get("mlp_ratio", 4.0)
        head_dim = model_cfg.get("head_dim", hidden_size // max(1, num_heads))
        
        mlp_dim = int(hidden_size * mlp_ratio)
        attn_dim = num_heads * head_dim
        
        embed_params = vocab_size * hidden_size
        layer_params = num_layers * (4 * hidden_size * attn_dim + 3 * hidden_size * mlp_dim)
        head_params = 2 * hidden_size * vocab_size
        param_count = embed_params + layer_params + head_params
        
    elif (checkpoint_dir / "config.json").exists():
        model_type = "hf_transformer"
        with open(checkpoint_dir / "config.json", "r") as f:
            cfg = json.load(f)
        
        hidden_size = cfg.get("hidden_size", 256)
        num_layers = cfg.get("num_hidden_layers", 6)
        vocab_size = cfg.get("vocab_size", 32000)
        intermediate_size = cfg.get("intermediate_size", hidden_size * 4)
        
        embed_params = vocab_size * hidden_size
        layer_params = num_layers * (12 * hidden_size * hidden_size + 3 * hidden_size * intermediate_size)
        head_params = hidden_size * vocab_size
        param_count = embed_params + layer_params + head_params
    
    if param_count is None:
        print("Error: Could not determine model parameters")
        sys.exit(1)
    
    bf16_gb = param_count * 2 / (1024 ** 3)
    fp16_gb = bf16_gb
    qlora_4bit_gb = param_count * 0.5 / (1024 ** 3)
    qlora_total_gb = qlora_4bit_gb + 0.5
    
    print(f"Model: {args.model_path} (type: {model_type})")
    print(f"Parameters: ~{param_count / 1e9:.2f}B ({param_count:,})")
    print()
    print("VRAM Estimates:")
    print(f"  Full finetuning (BF16/FP16): {bf16_gb:.2f} GB (weights) + ~{bf16_gb:.2f} GB (gradients) + ~{bf16_gb:.2f} GB (optimizer) = ~{bf16_gb * 3:.1f} GB")
    print(f"  LoRA (BF16 base + adapters): ~{bf16_gb:.2f} GB + small adapter overhead")
    print(f"  QLoRA (4-bit): ~{qlora_4bit_gb:.2f} GB (base) + ~0.5 GB (adapters, gradients) = ~{qlora_total_gb:.2f} GB")
    print()
    if args.use_qlora:
        print(f"=> With QLoRA: Expect {qlora_total_gb:.1f}-{qlora_total_gb * 1.5:.1f} GB VRAM for training")
    else:
        print(f"=> Without QLoRA: Consider enabling QLoRA for multi-billion parameter models")


def main() -> None:
    parser = argparse.ArgumentParser(description="Finetuning CLI helper")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export custom checkpoint to HF format")
    export_parser.add_argument("--input", "-i", type=str, required=True, help="Input checkpoint directory")
    export_parser.add_argument("--output", "-o", type=str, required=True, help="Output HF format directory")
    export_parser.add_argument("--model-type", type=str, default="nayhein_mini", help="Model type identifier")
    export_parser.add_argument("--force", action="store_true", help="Rebuild the HF export even if it already exists")
    export_parser.set_defaults(func=cmd_export)
    
    # Estimate command
    estimate_parser = subparsers.add_parser("estimate", help="Estimate VRAM requirements")
    estimate_parser.add_argument("--model-path", type=str, required=True, help="Model checkpoint path")
    estimate_parser.add_argument("--use-qlora", action="store_true", help="Show QLoRA-specific estimates")
    estimate_parser.set_defaults(func=cmd_estimate)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
