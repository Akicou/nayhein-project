
# Low-VRAM Finetuning Guide

## Overview

This update enables **QLoRA-based low-VRAM finetuning** for scaled custom models (like your 4B model). The key improvements:

1. **YAML-first configuration** - Single config files for SFT/DPO training
2. **True 4-bit QLoRA support** for custom DualModeModel checkpoints
3. **Automatic HF format export** - Converts custom checkpoints for bitsandbytes compatibility
4. **Memory safety warnings** - Alerts you when VRAM requirements are high

## Quick Start

### 1. Finetune your 4B scaled model with QLoRA

```bash
# Using the unified entrypoint with YAML config
python -m finetune --config config/finetune_4b_qlora.yaml
```

This will:
- Automatically export your custom checkpoint to a self-contained HF format with bundled remote code
- Automatically regenerate stale `hf_format` exports when the source checkpoint changes
- Load the model in 4-bit quantization (~0.5 bytes/param)
- Apply LoRA adapters to attention + MLP layers
- Use gradient checkpointing for activation memory savings

### 2. Alternative: Use CLI

```bash
# SFT with LoRA (no quantization)
python -m finetune.sft --model-path ./checkpoints/4b_scaled --use-lora --lora-r 16 --use-gradient-checkpointing

# QLoRA (4-bit) - RECOMMENDED for 4B models
python -m finetune.sft --model-path ./checkpoints/4b_scaled --use-lora --use-qlora --lora-r 16 --save-mode adapter
```

### 3. Export checkpoint manually (optional)

```bash
python tools/cli.py export --input ./checkpoints/4b_scaled --output ./checkpoints/4b_scaled_hf

# Force a rebuild if you want to replace an existing export manually
python tools/cli.py export --input ./checkpoints/4b_scaled --output ./checkpoints/4b_scaled_hf --force
```

### 4. Estimate VRAM requirements

```bash
python tools/cli.py estimate --model-path ./checkpoints/4b_scaled --use-qlora
```

## Configuration Files

### `config/finetune_4b_qlora.yaml` - Low-VRAM 4B SFT
- QLoRA 4-bit NF4 quantization
- LoRA rank 16 on attention + MLP
- Gradient accumulation for effective batch size
- Adapter-only saves

### `config/finetune_default.yaml` - Standard LoRA
- No quantization (requires more VRAM)
- Suitable for smaller models (<500M params)

### `config/finetune_dpo_qlora.yaml` - DPO with QLoRA
- Reference-free DPO for VRAM efficiency
- Same QLoRA setup as SFT

## VRAM Estimates

| Model Size | Full BF16 | QLoRA 4-bit | With Gradient Checkpointing |
|------------|-----------|-------------|----------------------------|
| 30M        | ~0.2 GB   | ~0.02 GB    | ~0.1 GB                    |
| 500M       | ~3 GB     | ~0.5 GB     | ~1 GB                      |
| 4B         | ~24 GB    | ~2.5 GB     | ~3-5 GB                    |

**Note:** Your 4B model went from ~70GB to **~3-5 GB** with QLoRA + gradient checkpointing.

## How It Works

### Before (70+ GB for 4B):
1. Load full model in BF16: 4B × 2 bytes = 8 GB
2. Gradients: another 8 GB
3. Optimizer states (Adam): 16-32 GB
4. Activations: 10-20 GB depending on seq length
5. Custom LoRA wrapper didn't reduce base model precision

### After (3-5 GB for 4B):
1. 4-bit quantized base: 4B × 0.5 bytes = 2.5 GB
2. LoRA adapters only: ~50-100 MB
3. Gradients for adapters: small
4. Optimizer states for adapters only: small
5. Gradient checkpointing: 50-70% activation savings

## Key Files Modified/Created

| File | Purpose |
|------|---------|
| `finetune/config.py` | YAML config loading + CLI override |
| `finetune/__main__.py` | Unified training entrypoint |
| `finetune/sft.py` | QLoRA support for custom checkpoints |
| `finetune/base.py` | Memory safety warnings |
| `finetune/utils.py` | Unified model loader with QLoRA |
| `tools/export_hf_format.py` | Custom → HF converter |
| `tools/cli.py` | Utility CLI (export, estimate) |
| `config/finetune_4b_qlora.yaml` | Example 4B QLoRA config |
| `config/finetune_dpo_qlora.yaml` | Example DPO QLoRA config |

## Tips

1. **Small batch size + high gradient accumulation** = memory efficiency
   ```yaml
   batch_size: 1
   gradient_accumulation_steps: 16  # effective batch = 16
   ```

2. **Adapter-only saves** save disk space during training:
   ```yaml
   save_mode: adapter
   ```

3. **Target all projection layers** for best LoRA coverage:
   ```yaml
   lora_target_modules:
     - q_proj
     - k_proj
     - v_proj
     - o_proj
     - gate_proj
     - up_proj
     - down_proj
   ```

4. **Monitor VRAM** with `nvidia-smi` or task manager during first run

## Troubleshooting

### "CUDA out of memory"
- Lower `batch_size` to 1
- Increase `gradient_accumulation_steps`
- Reduce `max_seq_length`
- Ensure `use_qlora: true` for models >1B params

### "QLoRA requires a 4-bit loaded model"
- Automatic with YAML configs
- If using CLI: add `--use-qlora` flag
- First run exports to HF format (takes a few minutes)

### Custom model loading issues
- Ensure checkpoint has `model.pt` and `config.pt`
- Export manually first: `python tools/cli.py export -i ./path -o ./path_hf`
- The exported HF directory now includes `configuration_nayhein_mini.py` and `modeling_nayhein_mini.py`
- The HF export only contains the AR/CausalLM path; diffusion and MTP heads are intentionally skipped
- Then use the HF directory as model path

## API Usage

```python
from finetune.config import trainer_config_from_yaml
from finetune.sft import SFTTrainer

config = trainer_config_from_yaml('config/finetune_4b_qlora.yaml', TrainerConfig)
trainer = SFTTrainer(config)
trainer.train()
```

---

**Summary:** Your 4B model now fits in consumer GPU memory (8-12GB cards work) using QLoRA. The YAML config handles all complexity automatically.
