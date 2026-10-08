"""Z-score fusion of per-matcher song scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def zscore(x: np.ndarray) -> np.ndarray:
    """Z-score across songs. Non-finite scores (no possible match) are set to the
    worst finite score first so they rank last without poisoning the stats."""
    x = np.asarray(x, dtype=np.float64).copy()
    ok = np.isfinite(x)
    if not ok.any():
        return np.zeros_like(x)
    x[~ok] = x[ok].min()
    sd = x.std()
    if sd < 1e-12:
        return np.zeros_like(x)
    return (x - x.mean()) / sd


@dataclass
class Fused:
    order: np.ndarray                 # song indices, best first
    fused: np.ndarray                 # fused score per song
    z: dict[str, np.ndarray]          # per-matcher z per song
    offset_matcher: np.ndarray        # name of matcher whose offset is used, per song


def fuse(scores: dict[str, np.ndarray], weights: dict[str, float], prior: np.ndarray | None = None) -> Fused:
    names = list(scores)
    n = len(next(iter(scores.values())))
    z = {m: zscore(scores[m]) for m in names}
    fused = np.zeros(n)
    for m in names:
        fused += float(weights.get(m, 1.0)) * z[m]
    if prior is not None:
        fused += prior
    Z = np.stack([z[m] for m in names], axis=1) if names else np.zeros((n, 0))
    off = np.array([names[i] for i in np.argmax(Z, axis=1)]) if names else np.array([""] * n)
    order = np.argsort(-fused, kind="stable")
    return Fused(order=order, fused=fused, z=z, offset_matcher=off)
