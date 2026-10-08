"""Matcher protocol and shared result type."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from ..features import Features


@dataclass
class MatchResult:
    score: float        # higher = better
    offset_s: float     # where in the song the best match starts
    detail: dict = field(default_factory=dict)  # matcher-specific extras for `explain`


NO_MATCH = MatchResult(score=-np.inf, offset_s=0.0)


@runtime_checkable
class Matcher(Protocol):
    name: str

    def score(self, query: Features, song: Features) -> MatchResult: ...


def score_many(matcher: Matcher, query: Features, songs: Sequence[Features]) -> list[MatchResult]:
    """Score one query against many song features. Matchers may provide a faster
    `score_many` method (e.g. coarse-to-fine search); otherwise loop."""
    fn = getattr(matcher, "score_many", None)
    if fn is not None:
        return fn(query, songs)
    return [matcher.score(query, s) for s in songs]


def voiced(pitch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(values, original_indices) of the finite frames of `pitch`."""
    idx = np.flatnonzero(np.isfinite(pitch))
    return np.ascontiguousarray(pitch[idx], dtype=np.float64), idx
