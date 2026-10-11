"""query -> ranked results over a library."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .config import Section
from .features import Features
from .fusion import fuse
from .matchers.base import MatchResult, score_many

MATCHER_NAMES = ("dtw", "contour", "rank")


def make_matchers(cfg: Section, names: Sequence[str] = MATCHER_NAMES) -> list:
    from .matchers.contour import ContourMatcher
    from .matchers.dtw import DTWMatcher
    from .matchers.rank import RankMatcher

    table = {"dtw": DTWMatcher, "contour": ContourMatcher, "rank": RankMatcher}
    return [table[n](cfg) for n in names]


@dataclass
class Entry:
    """A searchable song: metadata plus per-stem features."""
    song_id: str
    title: str
    artist: str
    stems: dict[str, Features]
    added_at: str | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class Result:
    rank: int
    entry: Entry
    fused: float
    z: dict[str, float]
    raw: dict[str, float]
    offset_s: float
    offset_matcher: str
    per_matcher: dict[str, MatchResult]   # best stem's result per matcher
    stem: dict[str, str]                  # which stem won, per matcher


@dataclass
class SearchOutput:
    results: list[Result]                 # all songs, ranked by fused score
    matcher_order: dict[str, np.ndarray]  # per-matcher ranking (entry indices, best first)


def artist_priors(entries: Sequence[Entry], artist: str | None, cfg: Section) -> dict[str, float]:
    if not artist:
        return {}
    from rapidfuzz import fuzz

    out = {}
    for e in entries:
        if fuzz.token_set_ratio(artist.lower(), e.artist.lower()) >= cfg.priors.artist_threshold:
            out[e.song_id] = float(cfg.priors.artist_boost)
    return out


def search(query: Features, entries: Sequence[Entry], cfg: Section,
           matchers: Sequence[str] = MATCHER_NAMES, priors: dict[str, float] | None = None,
           weights: dict[str, float] | None = None) -> SearchOutput:
    weights = dict(cfg.fusion.weights) if weights is None else weights
    items: list[Features] = []
    groups: list[int] = []
    stem_names: list[str] = []
    for gi, e in enumerate(entries):
        for stem, f in e.stems.items():
            items.append(f)
            groups.append(gi)
            stem_names.append(stem)

    n = len(entries)
    raw: dict[str, np.ndarray] = {}
    best_res: dict[str, list[MatchResult | None]] = {}
    best_stem: dict[str, list[str]] = {}
    for m in make_matchers(cfg, matchers):
        if m.name == "dtw":
            res = m.score_many(query, items, groups=groups)
        else:
            res = score_many(m, query, items)
        sc = np.full(n, -np.inf)
        br: list[MatchResult | None] = [None] * n
        bs = [""] * n
        for r, g, st in zip(res, groups, stem_names):
            if br[g] is None or r.score > sc[g]:
                sc[g] = r.score
                br[g] = r
                bs[g] = st
        raw[m.name], best_res[m.name], best_stem[m.name] = sc, br, bs

    prior = np.zeros(n)
    for i, e in enumerate(entries):
        prior[i] = (priors or {}).get(e.song_id, 0.0)
    fz = fuse(raw, weights, prior)

    results = []
    for rank, i in enumerate(fz.order, start=1):
        om = str(cfg.fusion.get("offset_matcher", ""))
        r = best_res[om][i] if om in best_res else None
        if r is None or not np.isfinite(r.score):
            om = str(fz.offset_matcher[i])
            r = best_res[om][i]
        results.append(Result(
            rank=rank, entry=entries[i], fused=float(fz.fused[i]),
            z={m: float(fz.z[m][i]) for m in raw}, raw={m: float(raw[m][i]) for m in raw},
            offset_s=float(r.offset_s) if r is not None else 0.0, offset_matcher=om,
            per_matcher={m: best_res[m][i] for m in raw}, stem={m: best_stem[m][i] for m in raw},
        ))
    order = {m: np.argsort(-np.where(np.isfinite(raw[m]), raw[m], -1e18), kind="stable") for m in raw}
    return SearchOutput(results=results, matcher_order=order)
