# Fix for NameError: name 'trainer_config_from_yaml' is not defined

## Problem
Running `python -m finetune --config config/finetune_4b_qlora.yaml` fails with:
```
NameError: name 'trainer_config_from_yaml' is not defined
```

## Solution
Your `finetune/main.py` is missing imports from the new `finetune.config` module.

### Quick Fix (Ubuntu)

Run these commands in your terminal:

```bash
# Backup existing main.py
cp finetune/main.py finetune/main.py.backup

# Apply the fix (add missing imports)
sed -i '4a from finetune.config import resolve_finetune_type, override_config_from_args, trainer_config_from_yaml' finetune/main.py
```

Or manually edit `finetune/main.py` and add this line after the other imports (around line 4-6):

```python
from finetune.config import resolve_finetune_type, override_config_from_args, trainer_config_from_yaml
```

### Complete Replacement

Alternatively, replace your entire `finetune/main.py` with the fixed version in `fix_finetune_main.py`:

```bash
# Backup existing
cp finetune/main.py finetune/main.py.backup

# Replace with fixed version
cp fix_finetune_main.py finetune/main.py
```

## Verify Fix

After applying the fix, test with:

```bash
python -m finetune --config config/finetune_4b_qlora.yaml
```

Should now work without the NameError.

## Why This Happened

Python's module loading priority:
1. `module/main.py` (your existing file)
2. `module/__main__.py` (new file I created)

Since `finetune/main.py` exists, Python runs it instead of `finetune/__main__.py`. The old `main.py` doesn't have the new `finetune.config` imports.
