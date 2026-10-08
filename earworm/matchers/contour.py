"""Matcher B: quantized interval contour + rhythm tokens, semi-global alignment."""

from __future__ import annotations

import numba as nb
import numpy as np

from ..config import Section
from ..features import Features
from .base import NO_MATCH, MatchResult

# interval classes, ordered so adjacent classes differ by 1
D_LARGE, D_SMALL, REPEAT, U_SMALL, U_LARGE = 0, 1, 2, 3, 4
INTERVAL_SYMBOLS = "DdRuU"
SHORTER, SAME, LONGER = 0, 1, 2
RHYTHM_SYMBOLS = "<=>"


def tokenize(notes: np.ndarray, c: Section) -> np.ndarray:
    """notes[n,3] -> int64[n-1, 2] of (interval_class, rhythm_class)."""
    if len(notes) < 2:
        return np.zeros((0, 2), dtype=np.int64)
    p = notes[:, 2].astype(np.float64)
    dur = np.maximum(notes[:, 1].astype(np.float64), 1e-6)
    d = np.diff(p)
    a = np.abs(d)
    ic = np.where(a < c.repeat_threshold, REPEAT,
                  np.where(a < c.large_threshold,
                           np.where(d > 0, U_SMALL, D_SMALL),
                           np.where(d > 0, U_LARGE, D_LARGE)))
    r = np.log2(dur[1:] / dur[:-1])
    rc = np.where(r < -c.rhythm_threshold, SHORTER, np.where(r > c.rhythm_threshold, LONGER, SAME))
    return np.stack([ic, rc], axis=1).astype(np.int64)


def tokens_to_str(tokens: np.ndarray) -> str:
    return " ".join(INTERVAL_SYMBOLS[i] + RHYTHM_SYMBOLS[r] for i, r in tokens)


def str_to_tokens(s: str) -> np.ndarray:
    toks = [(INTERVAL_SYMBOLS.index(t[0]), RHYTHM_SYMBOLS.index(t[1])) for t in s.split()]
    return np.array(toks, dtype=np.int64).reshape(-1, 2)


@nb.njit(cache=True)
def _sub_cost(a0, a1, b0, b1, adj, far, rmis):
    di = abs(a0 - b0)
    c = 0.0 if di == 0 else (adj if di == 1 else far)
    if a1 != b1:
        c += rmis
    return c


@nb.njit(cache=True)
def semiglobal_align(q, s, adj, far, rmis, indel):
    """Query fully aligned, song free at both ends. Returns (cost, start_j, end_j)
    where song tokens s[start_j:end_j] are the aligned region."""
    n = q.shape[0]
    m = s.shape[0]
    prev = np.zeros(m + 1)
    prev_st = np.arange(m + 1)
    cur = np.zeros(m + 1)
    cur_st = np.zeros(m + 1, np.int64)
    for i in range(1, n + 1):
        cur[0] = i * indel
        cur_st[0] = 0
        for j in range(1, m + 1):
            best = prev[j - 1] + _sub_cost(q[i - 1, 0], q[i - 1, 1], s[j - 1, 0], s[j - 1, 1], adj, far, rmis)
            st = prev_st[j - 1]
            v = prev[j] + indel          # query token unmatched
            if v < best:
                best = v
                st = prev_st[j]
            v = cur[j - 1] + indel       # song token unmatched
            if v < best:
                best = v
                st = cur_st[j - 1]
            cur[j] = best
            cur_st[j] = st
        for j in range(m + 1):
            prev[j] = cur[j]
            prev_st[j] = cur_st[j]
    bj = 0
    best = prev[0]
    for j in range(1, m + 1):
        if prev[j] < best:
            best = prev[j]
            bj = j
    return best, prev_st[bj], bj


class ContourMatcher:
    name = "contour"

    def __init__(self, cfg: Section):
        self.cfg = cfg
        self.c = cfg.contour

    def tokens(self, f: Features) -> np.ndarray:
        key = (self.name, "tokens")
        if key not in f.cache:
            f.cache[key] = tokenize(f.notes, self.c)
        return f.cache[key]

    def score(self, query: Features, song: Features) -> MatchResult:
        q = self.tokens(query)
        s = self.tokens(song)
        if len(q) == 0 or len(s) == 0:
            return NO_MATCH
        c = self.c
        cost, st, en = semiglobal_align(q, s, c.sub_adjacent, c.sub_far, c.rhythm_mismatch, c.indel)
        st = min(int(st), len(song.notes) - 1)
        # token k spans notes k and k+1; the region starts at note `st`
        last = min(max(int(en), st), len(song.notes) - 1)
        end_s = float(song.notes[last, 0] + song.notes[last, 1])
        return MatchResult(score=-(cost / len(q)), offset_s=float(song.notes[st, 0]),
                           detail={"start_note": st, "end_note": last, "end_s": end_s})
