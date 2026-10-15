"""Chargement de la configuration de l'experience."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Config:
    """Acces par chemin pointe : ``cfg.get("model.lightgbm.num_leaves")``."""

    def __init__(self, data: dict, path: Path):
        self._data = data
        self.path = path

    def get(self, dotted: str, default: Any = "__raise__") -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            elif default != "__raise__":
                return default
            else:
                raise KeyError(f"Cle de configuration absente : {dotted!r}")
        return node

    def section(self, dotted: str) -> dict:
        node = self.get(dotted)
        if not isinstance(node, dict):
            raise TypeError(f"{dotted!r} n'est pas une section")
        return node

    # ---- chemins utiles --------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / str(self.get("paths.data_dir"))

    @property
    def outputs_dir(self) -> Path:
        return PROJECT_ROOT / str(self.get("paths.outputs_dir"))

    @property
    def models_dir(self) -> Path:
        return PROJECT_ROOT / str(self.get("paths.models_dir"))

    def output(self, *parts: str) -> Path:
        path = self.outputs_dir.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def step_hours(self) -> float:
        return float(self.get("grid.interval_minutes")) / 60.0

    @property
    def random_state(self) -> int:
        return int(self.get("model.random_state"))


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh), cfg_path)
