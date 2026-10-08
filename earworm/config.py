"""Load config.toml into an attribute-accessible tree.

Lookup order: explicit path, $EARWORM_CONFIG, ./config.toml, then the
config.toml at the repository root (next to the package).
"""

from __future__ import annotations

import copy
import os
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent


class Section(dict):
    """dict with attribute access; nested tables become Sections."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(f"config has no key {key!r}") from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __deepcopy__(self, memo):
        return Section({k: copy.deepcopy(v, memo) for k, v in self.items()})


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Section({k: _wrap(v) for k, v in obj.items()})
    return obj


def find_config_path(path: str | os.PathLike | None = None) -> Path:
    candidates = []
    if path is not None:
        candidates.append(Path(path))
    if os.environ.get("EARWORM_CONFIG"):
        candidates.append(Path(os.environ["EARWORM_CONFIG"]))
    candidates += [Path.cwd() / "config.toml", _REPO_ROOT / "config.toml"]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(f"config.toml not found (tried: {', '.join(map(str, candidates))})")


def load_config(path: str | os.PathLike | None = None) -> Section:
    p = find_config_path(path)
    with open(p, "rb") as f:
        cfg = _wrap(tomllib.load(f))
    data_dir = Path(cfg.paths.data_dir)
    if not data_dir.is_absolute():
        data_dir = p.parent / data_dir
    cfg.paths.data_dir = str(data_dir)
    cfg.paths.config_file = str(p)
    return cfg


@lru_cache(maxsize=1)
def get_config() -> Section:
    """Process-wide default config. Callers that tweak values should deepcopy it."""
    return load_config()


def data_path(cfg: Section, *parts: str) -> Path:
    p = Path(cfg.paths.data_dir).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def resolve_device(cfg: Section) -> str:
    import torch

    prefer = cfg.device.prefer
    if prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
