import numpy as np
import pytest

from earworm.evaluation import (HumLabel, HumResult, find_song, fused_rank, leave_one_out, metrics,
                                pessimistic_rank, wilson_interval)
from earworm.search import Entry


def test_pessimistic_rank_counts_ties_against_truth():
    s = np.array([0.5, 0.9, 0.9, 0.1])
    assert pessimistic_rank(s, 1) == 2          # tied with song 2
    assert pessimistic_rank(s, 0) == 3
    assert pessimistic_rank(np.array([-np.inf, -np.inf, 0.0]), 0) == 3   # no possible match ranks last


def _res(z, truth=0):
    lab = HumLabel("h.wav", "t", "a", "chorus", "y")
    return HumResult(lab, "s0", {}, {m: np.asarray(v, float) for m, v in z.items()}, truth)


def test_fused_rank_ties_are_pessimistic():
    r = _res({"dtw": [1.0, 1.0], "contour": [0.0, 0.0], "rank": [0.0, 0.0]})
    assert fused_rank(r, {"dtw": 1, "contour": 1, "rank": 1}) == 2


def test_leave_one_out_dropping_an_uninformative_matcher_changes_nothing():
    rng = np.random.default_rng(0)
    results = []
    for _ in range(10):
        dtw = rng.normal(size=20)
        dtw[0] = 3.0                              # dtw always ranks the truth first
        results.append(_res({"dtw": dtw, "contour": rng.normal(size=20), "rank": np.zeros(20)}))
    table, paired = leave_one_out(results, {"dtw": 1.0, "contour": 1.0, "rank": 1.0})
    assert set(table) == {"all three", "contour+rank", "dtw+rank", "dtw+contour"}
    # rank is all zeros: dropping it changes no ranks
    assert paired["dtw+contour"]["better"] == paired["dtw+contour"]["worse"] == 0
    assert table["dtw+contour"]["mrr"] == table["all three"]["mrr"]
    # dropping dtw (the only informative matcher) can only hurt
    assert paired["contour+rank"]["better"] == 0
    assert table["contour+rank"]["mrr"] < table["all three"]["mrr"]


def test_find_song_flags_ambiguous_versions():
    entries = [Entry("a", "Hotline Bling", "Drake", {}), Entry("b", "Hotline Bling (Remix)", "Drake", {}),
               Entry("c", "Billie Jean", "Michael Jackson", {})]
    i, note = find_song(HumLabel("h", "Hotline Bling", "Drake", "chorus", "y"), entries)
    assert i == 0 and "AMBIGUOUS" in note and "Remix" in note
    i, note = find_song(HumLabel("h", "Billie Jean", "Michael Jackson", "chorus", "y"), entries)
    assert i == 2 and "AMBIGUOUS" not in note
    i, note = find_song(HumLabel("h", "Yesterday", "The Beatles", "chorus", "y"), entries)
    assert i is None and "scored" in note


def test_wilson_interval_and_metrics():
    lo, hi = wilson_interval(15, 30)
    assert lo == pytest.approx(0.332, abs=0.002) and hi == pytest.approx(0.668, abs=0.002)
    m = metrics([1, 2, 6, 10])
    assert m["top5"] == 0.5 and m["top5_lo"] < 0.5 < m["top5_hi"]
