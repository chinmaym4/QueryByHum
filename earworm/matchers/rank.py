"""Matcher C: rank contour (Kendall's tau between time-normalized pitch samples).

Works on voiced pitch_50ms frames only, for both query and song: windows are
measured in voiced time so that D (the query's voiced duration) and the window
length D*s are in the same units.
"""

from __future__ import annotations

import numba as nb
import numpy as np

from ..config import Section
from ..features import Features
from .base import NO_MATCH, MatchResult, voiced


def _signs(d: np.ndarray, tie_tolerance: float) -> np.ndarray:
    s = np.sign(d)
    if tie_tolerance > 0:
        s[np.abs(d) < tie_tolerance] = 0.0
    return s


def kendall_tau_b(x: np.ndarray, Y: np.ndarray, tie_tolerance: float = 0.0) -> np.ndarray:
    """Kendall's tau-b between vector x[L] and each row of Y[k, L], vectorized.

    Pairs whose values differ by less than `tie_tolerance` count as ties
    (tie_tolerance=0 is the textbook tau-b, as in scipy.stats.kendalltau)."""
    x = np.asarray(x, dtype=np.float64)
    Y = np.atleast_2d(np.asarray(Y, dtype=np.float64))
    i, j = np.triu_indices(x.shape[0], k=1)
    sx = _signs(x[i] - x[j], tie_tolerance)
    sy = _signs(Y[:, i] - Y[:, j], tie_tolerance)
    num = sy @ sx
    den = np.sqrt(np.count_nonzero(sx) * np.count_nonzero(sy, axis=1).astype(np.float64))
    with np.errstate(invalid="ignore", divide="ignore"):
        tau = np.where(den > 0, num / den, 0.0)
    return tau


@nb.njit(cache=True, parallel=True)
def _scan_windows(qs, sv, starts, offs, tol):
    """tau-b of qs against sv[start + offs] for every start; same math as
    kendall_tau_b, without materializing the (windows x pairs) arrays."""
    L = qs.shape[0]
    nq = 0
    for i in range(L):
        for j in range(i + 1, L):
            d = qs[i] - qs[j]
            if abs(d) >= tol and d != 0.0:
                nq += 1
    out = np.zeros(starts.shape[0])
    for k in nb.prange(starts.shape[0]):
        st = starts[k]
        num = 0.0
        ny = 0
        for i in range(L):
            yi = sv[st + offs[i]]
            xi = qs[i]
            for j in range(i + 1, L):
                dy = yi - sv[st + offs[j]]
                if abs(dy) < tol or dy == 0.0:
                    continue
                ny += 1
                dx = xi - qs[j]
                if abs(dx) < tol or dx == 0.0:
                    continue
                if (dx > 0) == (dy > 0):
                    num += 1.0
                else:
                    num -= 1.0
        if nq > 0 and ny > 0:
            out[k] = num / np.sqrt(nq * ny)
    return out


def sample_points(n: int, L: int) -> np.ndarray:
    return np.round(np.linspace(0, n - 1, L)).astype(np.int64)


class RankMatcher:
    name = "rank"

    def __init__(self, cfg: Section):
        self.cfg = cfg
        self.c = cfg.rank

    def _voiced(self, f: Features):
        key = (self.name, "voiced")
        if key not in f.cache:
            f.cache[key] = voiced(f.pitch_50ms)
        return f.cache[key]

    def windows(self, song: Features, nq: int):
        """Yield (start_frames, window_len) per scale, in voiced-frame units."""
        sv, _ = self._voiced(song)
        hop = max(1, int(round(self.c.hop_s / song.hop_50ms)))
        L = int(self.c.n_points)
        for s in self.c.scales:
            w = max(L, int(round(nq * s)))
            if sv.size < w:
                if sv.size < L:
                    continue
                w = sv.size
            starts = np.arange(0, sv.size - w + 1, hop, dtype=np.int64)
            yield starts, w

    def score(self, query: Features, song: Features) -> MatchResult:
        """Grid pass (hop_s, scale grid), then refine the best `refine_windows`
        windows at a 1-frame start step and scales +/- refine_scale_step: tau is
        sensitive to start shifts of a few frames, so the grid alone can miss the
        true alignment and pick a compromise window elsewhere."""
        qv, _ = self._voiced(query)
        sv, sidx = self._voiced(song)
        L = int(self.c.n_points)
        if qv.size < L or sv.size < L:
            return NO_MATCH
        qs = qv[sample_points(qv.size, L)]
        tol = float(self.c.get("tie_tolerance", 0.0))
        cands = []   # (tau, start, window_frames, scale)
        for s, (starts, w) in zip(self.c.scales, self.windows(song, qv.size)):
            tau = _scan_windows(qs, sv, starts, sample_points(w, L), tol)
            cands += [(float(tau[k]), int(starts[k]), w, s) for k in np.argsort(-tau)[: int(self.c.refine_windows)]]
        if not cands:
            return NO_MATCH
        cands.sort(reverse=True)
        best = cands[0]
        hop = max(1, int(round(self.c.hop_s / song.hop_50ms)))
        ds = float(self.c.refine_scale_step)
        for _, st, _, s in cands[: int(self.c.refine_windows)]:
            for s2 in (s - ds, s, s + ds):
                w = min(sv.size, max(L, int(round(qv.size * s2))))
                starts = np.arange(max(0, st - hop + 1), min(sv.size - w, st + hop - 1) + 1, dtype=np.int64)
                if starts.size == 0:
                    continue
                tau = _scan_windows(qs, sv, starts, sample_points(w, L), tol)
                k = int(np.argmax(tau))
                if tau[k] > best[0]:
                    best = (float(tau[k]), int(starts[k]), w, s2)
        tau, start, w, scale = best
        hop_s = song.hop_50ms
        return MatchResult(score=tau, offset_s=float(sidx[start] * hop_s),
                           detail={"start_frame": start, "window_frames": w, "scale": scale,
                                   "end_s": float((sidx[start + w - 1] + 1) * hop_s)})
