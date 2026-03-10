from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from pretrain import DualModeModel
from tools.export_hf_format import export_dualmode_to_hf


def _load_custom_checkpoint_module():
    module_path = Path(__file__).resolve().parents[1] / "finetune" / "custom_checkpoint.py"
    spec = importlib.util.spec_from_file_location("test_custom_checkpoint", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CUSTOM_CHECKPOINT = _load_custom_checkpoint_module()


class FakeTokenizer:
    def save_pretrained(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        (output_path / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 128}), encoding="utf-8")
        (output_path / "special_tokens_map.json").write_text(
            json.dumps({"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>"}),
            encoding="utf-8",
        )
        (output_path / "vocab.json").write_text(json.dumps({"<pad>": 0, "<bos>": 1, "<eos>": 2}), encoding="utf-8")
        (output_path / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")


class CustomHFExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.mkdtemp(prefix="nayhein_export_test_")
        self.checkpoint_dir = Path(self.tempdir) / "checkpoint"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tempdir, ignore_errors=True)

    def _write_checkpoint(self, *, hidden_size: int, num_layers: int, num_heads: int, head_dim: int, max_seq_len: int = 16) -> None:
        model = DualModeModel(
            vocab_size=15,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            mlp_ratio=4.0,
            max_seq_len=max_seq_len,
            mtp_enabled=False,
        )
        torch.save(model.state_dict(), self.checkpoint_dir / "model.pt")
        torch.save(
            {
                "vocab_size": model.vocab_size,
                "original_vocab_size": model.original_vocab_size,
                "mask_token_id": model.mask_token_id,
                "hidden_size": model.hidden_size,
                "num_layers": model.num_layers,
                "num_heads": model.num_heads,
                "head_dim": model.head_dim,
                "max_seq_len": model.max_seq_len,
                "mlp_ratio": model.mlp_ratio,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "pad_token_id": 0,
            },
            self.checkpoint_dir / "config.pt",
        )

    def _write_tokenizer_files(self, path: Path) -> None:
        FakeTokenizer().save_pretrained(path)

    def test_export_writes_self_contained_hf_dir(self) -> None:
        self._write_checkpoint(hidden_size=12, num_layers=2, num_heads=3, head_dim=4)
        self._write_tokenizer_files(self.checkpoint_dir)
        export_dir = self.checkpoint_dir / "hf_export_files"

        export_dualmode_to_hf(self.checkpoint_dir, export_dir)

        expected = {
            "config.json",
            "pytorch_model.bin",
            "configuration_nayhein_mini.py",
            "modeling_nayhein_mini.py",
            "nayhein_export.json",
            "tokenizer_config.json",
        }
        self.assertTrue(expected.issubset({p.name for p in export_dir.iterdir()}))

    def test_auto_config_and_auto_model_load_with_remote_code(self) -> None:
        self._write_checkpoint(hidden_size=12, num_layers=2, num_heads=3, head_dim=4)
        self._write_tokenizer_files(self.checkpoint_dir)
        export_dir = self.checkpoint_dir / "hf_export_auto_model"

        export_dualmode_to_hf(self.checkpoint_dir, export_dir)

        config = AutoConfig.from_pretrained(export_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(export_dir, trust_remote_code=True)

        self.assertEqual(config.__class__.__name__, "NayheinMiniConfig")
        self.assertEqual(model.__class__.__name__, "NayheinMiniForCausalLM")

    def test_forward_pass_matches_expected_shape(self) -> None:
        self._write_checkpoint(hidden_size=12, num_layers=2, num_heads=3, head_dim=4)
        self._write_tokenizer_files(self.checkpoint_dir)
        export_dir = self.checkpoint_dir / "hf_export_forward"

        export_dualmode_to_hf(self.checkpoint_dir, export_dir)
        model = AutoModelForCausalLM.from_pretrained(export_dir, trust_remote_code=True)

        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        outputs = model(input_ids=input_ids)
        self.assertEqual(tuple(outputs.logits.shape), (1, 4, model.config.vocab_size))

    def test_odd_head_dim_is_supported(self) -> None:
        self._write_checkpoint(hidden_size=15, num_layers=2, num_heads=3, head_dim=5)
        self._write_tokenizer_files(self.checkpoint_dir)
        export_dir = self.checkpoint_dir / "hf_export_odd"

        export_dualmode_to_hf(self.checkpoint_dir, export_dir)
        model = AutoModelForCausalLM.from_pretrained(export_dir, trust_remote_code=True)

        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        outputs = model(input_ids=input_ids)
        self.assertEqual(tuple(outputs.logits.shape), (1, 4, model.config.vocab_size))
        self.assertEqual(model.config.head_dim, 5)

    def test_stale_export_is_regenerated(self) -> None:
        self._write_checkpoint(hidden_size=12, num_layers=2, num_heads=3, head_dim=4)
        self._write_tokenizer_files(self.checkpoint_dir)
        export_dir = self.checkpoint_dir / "hf_format"

        export_dualmode_to_hf(self.checkpoint_dir, export_dir)
        metadata_path = export_dir / "nayhein_export.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["export_version"] = 0
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

        resolved = CUSTOM_CHECKPOINT.ensure_hf_export(self.checkpoint_dir)
        refreshed = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.assertEqual(resolved, export_dir)
        self.assertEqual(refreshed["export_version"], CUSTOM_CHECKPOINT.EXPORT_VERSION)

    def test_missing_tokenizer_files_use_fallback_tokenizer(self) -> None:
        self._write_checkpoint(hidden_size=12, num_layers=2, num_heads=3, head_dim=4)
        export_dir = self.checkpoint_dir / "hf_format"

        with mock.patch("tools.export_hf_format.get_default_tokenizer", return_value=FakeTokenizer()):
            export_dualmode_to_hf(self.checkpoint_dir, export_dir)

        self.assertTrue((export_dir / "tokenizer_config.json").exists())
        self.assertTrue((export_dir / "vocab.json").exists())
        self.assertTrue((export_dir / "merges.txt").exists())


if __name__ == "__main__":
    unittest.main()
