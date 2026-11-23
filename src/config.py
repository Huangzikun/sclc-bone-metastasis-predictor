"""读 config.yaml，相对路径都以仓库根目录为基准"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

def _default_root() -> Path:
    return Path(__file__).resolve().parent.parent

class Config:
    def __init__(self, cfg: dict, root: Path):
        self._cfg = cfg
        self.root = root

    @property
    def raw(self) -> dict:
        return self._cfg

    def get(self, *keys, default=None):
        node = self._cfg
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    def resolve(self, path: str | None):
        if path is None:
            return None
        p = Path(path)
        return p if p.is_absolute() else (self.root / path)

    @property
    def artifacts_dir(self) -> Path:
        return self.resolve(self.get("paths", "artifacts_dir"))

    def ensure_artifacts(self) -> Path:
        d = self.artifacts_dir
        d.mkdir(parents=True, exist_ok=True)
        return d

def load_config(config_path: str | os.PathLike | None = None) -> Config:
    if config_path is None:
        here = Path(__file__).resolve().parent
        config_path = here.parent / "config.yaml"
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    root = os.environ.get("BM_PIPELINE_ROOT")
    root = Path(root) if root else _default_root()
    return Config(data, root)
