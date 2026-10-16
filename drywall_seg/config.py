"""Config = YAML file (+ optional `_base_` inheritance) + dotted CLI overrides.

    cfg = load_config("configs/clipseg_ft.yaml", ["train.epochs=5", "model.freeze_vision=false"])
    cfg.train.epochs            # 5
"""
from __future__ import annotations

import copy
import re
from pathlib import Path

import yaml

# YAML 1.1 reads "1e-4" (no dot) as a string; treat anything shaped like a float as one.
_FLOAT = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


class Config(dict):
    """Nested dict with attribute access."""

    def __init__(self, data=None):
        super().__init__()
        for k, v in (data or {}).items():
            self[k] = Config(v) if isinstance(v, dict) else v

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self[key] = Config(value) if isinstance(value, dict) else value

    def to_dict(self):
        return {k: v.to_dict() if isinstance(v, Config) else v for k, v in self.items()}


def _coerce(value):
    if isinstance(value, dict):
        return {k: _coerce(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_coerce(v) for v in value]
    if isinstance(value, str) and _FLOAT.match(value.strip()):
        return float(value)
    return value


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read(path):
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = data.pop("_base_", None)
    return _merge(_read(path.parent / base), data) if base else data


def load_config(path, overrides=()):
    data = _read(path)
    for item in overrides:
        key, sep, text = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key.sub=value, got {item!r}")
        *parents, leaf = key.strip().split(".")
        node = data
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(text)
    return Config(_coerce(data))


def save_config(cfg, path):
    Path(path).write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")
