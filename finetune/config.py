from __future__ import annotations

from dataclasses import MISSING, asdict, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Type, TypeVar, get_args, get_origin, Union
import yaml

from .base import TrainerConfig
from .dpo import DPOConfig

T = TypeVar("T", TrainerConfig, DPOConfig)


def _merge_dicts(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _collect_yaml_sections(raw: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for section in ("model", "data", "training", "lora", "quantization", "save", "runtime", "dpo"):
        value = raw.get(section)
        if isinstance(value, Mapping):
            merged = _merge_dicts(merged, value)
    if isinstance(raw.get("finetune"), Mapping):
        merged = _merge_dicts(merged, raw["finetune"])
    return merged


def _normalize_value(field_type: Any, value: Any) -> Any:
    origin = get_origin(field_type)
    if origin is None:
        return value
    if origin is list and isinstance(value, str):
        return [value]
    if origin is Optional:
        subtypes = [arg for arg in get_args(field_type) if arg is not type(None)]
        if value is None or not subtypes:
            return value
        return _normalize_value(subtypes[0], value)
    if origin is Union:
        for t in get_args(field_type):
            if t is not type(None) and t is not type(Ellipsis):
                return t(value) if isinstance(value, str) else value
    return value


def _dataclass_defaults(cls: Type[T]) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}
    for field in fields(cls):
        if field.default is not MISSING:
            defaults[field.name] = field.default
        elif hasattr(field, "default_factory") and field.default_factory is not MISSING:  # Dataclass field has default_factory
            defaults[field.name] = field.default_factory()
    return defaults


def _build_config(config_cls: Type[T], values: Mapping[str, Any]) -> T:
    defaults = _dataclass_defaults(config_cls)
    allowed = {field.name: field for field in fields(config_cls)}
    init_values: Dict[str, Any] = {}
    for key, field in allowed.items():
        if key in values:
            init_values[key] = _normalize_value(field.type, values[key])
        elif key in defaults:
            init_values[key] = defaults[key]
    return config_cls(**init_values)


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("YAML config root must be a mapping")
    merged = _collect_yaml_sections(raw)
    merged["config_path"] = str(path)
    return merged


def trainer_config_from_yaml(config_path: str | Path, config_cls: Type[T]) -> T:
    values = load_yaml_config(config_path)
    return _build_config(config_cls, values)


def override_config_from_args(config: T, args: Any, skip: Optional[set[str]] = None) -> T:
    if not isinstance(config, object):
        raise TypeError("config must be a valid object")
    skip = set(skip or set())
    data = asdict(config)
    for field in fields(config):
        if field.name in skip or not hasattr(args, field.name):
            continue
        arg_value = getattr(args, field.name)
        if arg_value is not None:
            data[field.name] = arg_value
    return type(config)(**data)


def resolve_finetune_type(config_path: str | Path) -> str:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("YAML config root must be a mapping")
    finetune = raw.get("finetune", {})
    if isinstance(finetune, Mapping):
        return str(finetune.get("type", "sft")).lower()
    return "sft"
