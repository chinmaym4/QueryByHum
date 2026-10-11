"""Matcher A: subsequence DTW on continuous pitch (pitch_50ms, voiced frames only).

Each query frame contributes exactly one local cost, so accumulated cost / N is
the mean clipped pitch error along the path:
  (1,1): +c(i, j)
  (1,2): +c(i, j)            (song frame j-1 skipped)
  (2,1): +c(i-1, j) + c(i, j) (two query frames on one song frame)
This bounds the tempo ratio to [0.5, 2].
"""

from __future__ import annotations

from typing import Sequence

import numba as nb
import numpy as np

from ..config import Section
from ..features import Features
from ..pitch import downsample_nanmedian
from .base import NO_MATCH, MatchResult, voiced

_INF = 1e18


@nb.njit(cache=True, fastmath=True)
def _subseq_dtw(q, s, clip):
    """Returns (total_cost, start_j, end_j). q must be fully consumed; s has free ends."""
    n = q.shape[0]
    m = s.shape[0]
    d2 = np.full(m, _INF)   # row i-2
    d1 = np.full(m, _INF)   # row i-1
    d0 = np.full(m, _INF)   # row i
    s2 = np.zeros(m, np.int64)
    s1 = np.zeros(m, np.int64)
    s0 = np.zeros(m, np.int64)
    for j in range(m):
        d1[j] = min(abs(q[0] - s[j]), clip)
        s1[j] = j
    if n == 1:
        best = _INF
        bj = -1
        for j in range(m):
            if d1[j] < best:
                best = d1[j]
                bj = j
        return best, bj, bj
    for i in range(1, n):
        for j in range(m):
            c = min(abs(q[i] - s[j]), clip)
            best = _INF
            st = -1
            if j >= 1 and d1[j - 1] < best:
                best = d1[j - 1]
                st = s1[j - 1]
            if j >= 2 and d1[j - 2] < best:
                best = d1[j - 2]
                st = s1[j - 2]
            c2 = min(abs(q[i - 1] - s[j]), clip)
            if i == 1:
                # (2,1) step from the virtual row before the query: start at j
                if c2 < best:
                    best = c2
                    st = j
            elif j >= 1 and d2[j - 1] + c2 < best:
                best = d2[j - 1] + c2
                st = s2[j - 1]
            d0[j] = best + c
            s0[j] = st
        # rotate rows
        tmp = d2
        d2 = d1
        d1 = d0
        d0 = tmp
        tmps = s2
        s2 = s1
        s1 = s0
        s0 = tmps
    best = _INF
    bj = -1
    for j in range(m):
        if d1[j] < best:
            best = d1[j]
            bj = j
    if bj < 0:
        return _INF, -1, -1
    return best, s1[bj], bj


@nb.njit(cache=True, parallel=True)
def _dtw_batch(qc, concat, offsets, centers, shifts, clip):
    """For each song k (concat[offsets[k]:offsets[k+1]], centered by centers[k]),
    find the best transposition. Returns cost, start, end, shift arrays."""
    k_songs = offsets.shape[0] - 1
    cost = np.full(k_songs, _INF)
    start = np.full(k_songs, -1, np.int64)
    end = np.full(k_songs, -1, np.int64)
    shift = np.zeros(k_songs)
    for k in nb.prange(k_songs):
        a = offsets[k]
        b = offsets[k + 1]
        if b - a < 1:
            continue
        s = concat[a:b] - centers[k]
        for t in shifts:
            c, st, en = _subseq_dtw(qc + t, s, clip)
            if c < cost[k]:
                cost[k] = c
                start[k] = st
                end[k] = en
                shift[k] = t
    return cost, start, end, shift


@nb.njit(cache=True)
def _dtw_path(q, s, clip):
    """Full-matrix version with backtracking, for plotting one alignment."""
    n = q.shape[0]
    m = s.shape[0]
    D = np.full((n, m), _INF)
    P = np.zeros((n, m), np.int8)   # 0 start, 1 diag, 2 (1,2), 3 (2,1), 4 (2,1) from virtual row
    for j in range(m):
        D[0, j] = min(abs(q[0] - s[j]), clip)
    for i in range(1, n):
        for j in range(m):
            c = min(abs(q[i] - s[j]), clip)
            c2 = min(abs(q[i - 1] - s[j]), clip)
            best = _INF
            p = 0
            if j >= 1 and D[i - 1, j - 1] < best:
                best = D[i - 1, j - 1]
                p = 1
            if j >= 2 and D[i - 1, j - 2] < best:
                best = D[i - 1, j - 2]
                p = 2
            if i == 1:
                if c2 < best:
                    best = c2
                    p = 4
            elif j >= 1 and D[i - 2, j - 1] + c2 < best:
                best = D[i - 2, j - 1] + c2
                p = 3
            D[i, j] = best + c
            P[i, j] = p
    j = int(np.argmin(D[n - 1]))
    qi = []
    sj = []
    i = n - 1
    while i >= 0:
        qi.append(i)
        sj.append(j)
        p = P[i, j]
        if i == 0 or p == 0:
            break
        if p == 1:
            i -= 1
            j -= 1
        elif p == 2:
            i -= 1
            j -= 2
        elif p == 3:
            qi.append(i - 1)
            sj.append(j)
            i -= 2
            j -= 1
        else:
            qi.append(0)
            sj.append(j)
            break
    return np.array(qi[::-1]), np.array(sj[::-1])


@nb.njit(cache=True, parallel=True)
def _key_refine_batch(qc, concat, offsets, centers, t0, clip):
    """Fractional key correction. For each song k, take the DTW path at its
    current shift t0[k], set the shift to t0 + median(song - shifted query) over
    the aligned frame pairs, and rescore at that shift. Returns cost, start, end,
    shift of the rescored alignment (the caller keeps whichever is better)."""
    k_songs = offsets.shape[0] - 1
    cost = np.full(k_songs, _INF)
    start = np.full(k_songs, -1, np.int64)
    end = np.full(k_songs, -1, np.int64)
    shift = np.zeros(k_songs)
    for k in nb.prange(k_songs):
        a = offsets[k]
        b = offsets[k + 1]
        if b - a < 1:
            continue
        s = concat[a:b] - centers[k]
        q = qc + t0[k]
        qi, sj = _dtw_path(q, s, clip)
        diff = np.empty(qi.shape[0])
        for m in range(qi.shape[0]):
            diff[m] = s[sj[m]] - q[qi[m]]
        t = t0[k] + np.median(diff)
        c, st, en = _subseq_dtw(qc + t, s, clip)
        cost[k] = c
        start[k] = st
        end[k] = en
        shift[k] = t
    return cost, start, end, shift


def _shifts(lo: int, hi: int, step: int) -> np.ndarray:
    return np.arange(lo, hi + 1e-9, step, dtype=np.float64)


class DTWMatcher:
    name = "dtw"

    def __init__(self, cfg: Section):
        self.cfg = cfg
        self.c = cfg.dtw

    # -- per-features preprocessing (cached on the Features object) --
    def _prep(self, f: Features, level: str):
        key = (self.name, level)
        if key not in f.cache:
            p = f.pitch_50ms
            hop = f.hop_50ms
            if level == "coarse":
                p = downsample_nanmedian(p, int(self.c.coarse_factor))
                hop = hop * int(self.c.coarse_factor)
            vals, idx = voiced(p)
            center = float(np.median(vals)) if vals.size else 0.0
            f.cache[key] = (vals, idx, center, hop)
        return f.cache[key]

    def _run(self, query: Features, songs: Sequence[Features], level: str, shifts: np.ndarray):
        qv, _, qc, _ = self._prep(query, level)
        if qv.size < 2:
            return None
        preps = [self._prep(s, level) for s in songs]
        lens = np.array([p[0].size for p in preps], dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        concat = np.concatenate([p[0] for p in preps]) if preps else np.zeros(0)
        centers = np.array([p[2] for p in preps], dtype=np.float64)
        clip = float(self.c.cost_clip)
        cost, start, end, shift = _dtw_batch(qv - qc, concat, offsets, centers, shifts, clip)
        if level == "fine" and self.c.key_refine:
            ok = cost < _INF / 2
            c2, s2, e2, t2 = _key_refine_batch(qv - qc, concat, offsets, centers, shift, clip)
            better = ok & (c2 < cost)
            cost = np.where(better, c2, cost)
            start = np.where(better, s2, start)
            end = np.where(better, e2, end)
            shift = np.where(better, t2, shift)
        out = []
        for k, p in enumerate(preps):
            if cost[k] >= _INF / 2:
                out.append(NO_MATCH)
                continue
            _, idx, center, hop = p
            out.append(MatchResult(
                score=-(cost[k] / qv.size),
                offset_s=float(idx[start[k]] * hop),
                detail={"transpose": float(shift[k]), "song_center": center, "query_center": qc,
                        "end_s": float((idx[end[k]] + 1) * hop), "level": level},
            ))
        return out

    def score(self, query: Features, song: Features) -> MatchResult:
        out = self._run(query, [song], "fine", self._fine_shifts())
        return out[0] if out else NO_MATCH

    def _fine_shifts(self):
        return _shifts(self.c.transpose_min, self.c.transpose_max, self.c.transpose_step)

    def score_many(self, query: Features, songs: Sequence[Features], groups: Sequence | None = None) -> list[MatchResult]:
        """Coarse pass (100 ms, transpositions step 2) over everything, then full
        resolution on all items of the top `refine_top` groups (songs)."""
        if not songs:
            return []
        if not self.c.coarse:
            return self._run(query, songs, "fine", self._fine_shifts()) or [NO_MATCH] * len(songs)
        coarse = self._run(query, songs, "coarse",
                           _shifts(self.c.transpose_min, self.c.transpose_max, self.c.coarse_transpose_step))
        if coarse is None:
            return [NO_MATCH] * len(songs)
        groups = list(range(len(songs))) if groups is None else list(groups)
        best: dict = {}
        for g, r in zip(groups, coarse):
            best[g] = max(best.get(g, -np.inf), r.score)
        top = set(sorted(best, key=best.get, reverse=True)[: int(self.c.refine_top)])
        sel = [k for k, g in enumerate(groups) if g in top]
        fine = self._run(query, [songs[k] for k in sel], "fine", self._fine_shifts())
        out = list(coarse)
        for k, r in zip(sel, fine):
            out[k] = r
        return out

    def alignment(self, query: Features, song: Features, result: MatchResult):
        """(query_times, song_times, query_pitch_transposed, song_pitch) along the best path."""
        qv, qidx, qc, hop = self._prep(query, "fine")
        sv, sidx, sc, _ = self._prep(song, "fine")
        t = result.detail.get("transpose", 0.0)
        qi, sj = _dtw_path(qv - qc + t, sv - sc, float(self.c.cost_clip))
        return qidx[qi] * hop, sidx[sj] * hop, (qv - qc + t + sc)[qi], sv[sj]
