import numpy as np
import pytest

from types import SimpleNamespace

from earworm.evaluation import (HumLabel, HumResult, find_song, fused_rank, leave_one_out, matcher_offsets, metrics,
                                offset_accuracy, pessimistic_rank, read_labels, sample_playlist, spread_by_group,
                                wilson_interval)
from earworm.matchers.base import NO_MATCH, MatchResult
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


def test_read_labels_true_offset_optional_and_validated(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("hum_file,title,artist,part,in_preview,notes,true_offset_s\n"
                 "a.wav,T,A,chorus,y,,12.5\nb.wav,T,A,verse,y,,\n")
    labs = read_labels(p)
    assert labs[0].true_offset_s == 12.5 and labs[1].true_offset_s is None
    # old files without the column still load
    p.write_text("hum_file,title,artist,part,in_preview,notes\na.wav,T,A,chorus,y,\n")
    assert read_labels(p)[0].true_offset_s is None
    for bad in ("abc", "-3", "nan"):
        p.write_text(f"hum_file,title,artist,part,in_preview,notes,true_offset_s\na.wav,T,A,chorus,y,,{bad}\n")
        with pytest.raises(ValueError, match="true_offset_s"):
            read_labels(p)


def _pm(**offsets):
    return SimpleNamespace(per_matcher={m: (MatchResult(-0.1, o) if o is not None else NO_MATCH)
                                        for m, o in offsets.items()})


def test_matcher_offsets_and_max_spread():
    off, spread = matcher_offsets(_pm(dtw=8.0, contour=8.5, rank=10.25))
    assert off == {"dtw": 8.0, "contour": 8.5, "rank": 10.25} and spread == pytest.approx(2.25)
    off, spread = matcher_offsets(_pm(dtw=8.0, contour=None, rank=9.0))     # a matcher with no match is NaN
    assert np.isnan(off["contour"]) and spread == pytest.approx(1.0)
    assert np.isnan(matcher_offsets(_pm(dtw=8.0, contour=None, rank=None))[1])


def _off_res(true, ranks_fused, **offs):
    lab = HumLabel("h.wav", "t", "a", "chorus", "y", true_offset_s=true)
    r = HumResult(lab, "s0", {"dtw": 1, "contour": 1, "rank": 1, "fused": ranks_fused}, {}, 0)
    r.offsets = offs
    return r


def test_offset_accuracy_counts_missing_offsets_as_misses():
    rs = [_off_res(8.0, 1, dtw=8.2, contour=9.5, rank=float("nan"), fused=8.2),
          _off_res(5.0, 3, dtw=5.9, contour=5.0, rank=5.0, fused=7.0),
          _off_res(None, 1, dtw=0.0, contour=0.0, rank=0.0, fused=0.0)]          # no true offset: ignored
    t = offset_accuracy(rs, 1.0)
    assert t["dtw"]["n"] == 2 and t["dtw"]["within"] == 1.0
    assert t["contour"]["within"] == 0.5
    assert t["rank"]["within"] == 0.5 and t["rank"]["n_no_offset"] == 1       # NaN offset is a miss
    assert t["fused"]["within"] == 0.5 and t["fused"]["median_abs_err"] == pytest.approx(1.1)
    t1 = offset_accuracy(rs, 1.0, only_fused_top1=True)
    assert t1["dtw"]["n"] == 1 and t1["fused"]["within"] == 1.0


def test_sample_playlist_is_seeded_distinct_and_ordered():
    rows = [f"row{i}" for i in range(153)]
    a = sample_playlist(rows, 40, 42)
    assert a == sample_playlist(rows, 40, 42)                      # deterministic
    assert a != sample_playlist(rows, 40, 43)
    assert len({i for i, _ in a}) == 40 and [i for i, _ in a] == sorted(i for i, _ in a)
    assert all(rows[i - 1] == r for i, r in a)                      # 1-based row numbers line up
    with pytest.raises(ValueError):
        sample_playlist(rows, 154, 0)


def test_spread_by_group_splits_on_fused_first():
    def res(fused_rank, spread):
        r = _off_res(None, fused_rank)
        r.spread_s = spread
        return r
    rs = [res(1, 0.1), res(1, 2.0), res(1, 0.3), res(4, 5.0), res(9, float("nan")), res(2, 1.0)]
    g = spread_by_group(rs, 1.0)
    first, other = g["fused ranked song 1st"], g["all other hums"]
    assert first == {"hums": 3, "with_spread": 3, "median": 0.3, "max": 2.0, "over_tol": 1}
    assert other["hums"] == 3 and other["with_spread"] == 2          # NaN spread is counted as a hum, not a spread
    assert other["median"] == pytest.approx(3.0) and other["max"] == 5.0
    assert other["over_tol"] == 1                                    # 1.0 is not > 1.0
    empty = spread_by_group([], 1.0)["all other hums"]
    assert empty["hums"] == 0 and np.isnan(empty["median"]) and empty["over_tol"] == 0
