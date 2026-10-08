import numpy as np
import pytest
from scipy.stats import kendalltau

from earworm.features import features_from_pitch
from earworm.matchers.contour import _sub_cost, semiglobal_align, str_to_tokens, tokenize
from earworm.matchers.dtw import DTWMatcher, _subseq_dtw
from earworm.matchers.rank import kendall_tau_b
from earworm.fusion import fuse, zscore


# ---- contour ----
def _align(cfg, q, s):
    c = cfg.contour
    return semiglobal_align(str_to_tokens(q), str_to_tokens(s), c.sub_adjacent, c.sub_far, c.rhythm_mismatch, c.indel)


def test_contour_exact_substring_costs_zero(cfg):
    cost, st, en = _align(cfg, "u= d< R>", "D= u= d< R> U=")
    assert cost == 0.0 and (st, en) == (1, 4)


def test_contour_substitution_costs(cfg):
    assert _align(cfg, "u=", "U=")[0] == pytest.approx(0.5)   # adjacent interval classes
    assert _align(cfg, "u=", "d=")[0] == pytest.approx(1.0)   # u vs d: two classes apart
    assert _align(cfg, "u=", "u>")[0] == pytest.approx(0.3)   # rhythm only
    c = cfg.contour
    assert _sub_cost(4, 0, 0, 2, c.sub_adjacent, c.sub_far, c.rhythm_mismatch) == pytest.approx(1.3)
    # ... but a 1.3 substitution is never chosen over a 1.0 deletion
    assert _align(cfg, "U<", "D>")[0] == pytest.approx(1.0)


def test_contour_insertion_deletion(cfg):
    # an extra song token inside the match costs one indel
    assert _align(cfg, "u= u= U=", "u= D> u= U=")[0] == pytest.approx(1.0)
    # an extra query token costs one indel
    assert _align(cfg, "u= R= d= U=", "D< u= d= U= D<")[0] == pytest.approx(1.0)
    # query longer than song: unmatched query tokens all pay indel
    assert _align(cfg, "u= u= u=", "u=")[0] == pytest.approx(2.0)


def test_tokenize_classes(cfg):
    notes = np.array([[0, 0.5, 60], [0.5, 0.5, 60.2], [1, 0.25, 61.5], [1.25, 0.5, 65], [1.75, 0.5, 63.6]], np.float32)
    toks = tokenize(notes, cfg.contour)
    from earworm.matchers.contour import tokens_to_str
    assert tokens_to_str(toks) == "R= u< U> d="


# ---- rank ----
def test_kendall_tau_matches_scipy():
    rng = np.random.default_rng(0)
    for _ in range(50):
        x = rng.normal(size=24)
        Y = rng.normal(size=(7, 24))
        ours = kendall_tau_b(x, Y)
        ref = [kendalltau(x, y).statistic for y in Y]
        np.testing.assert_allclose(ours, ref, atol=1e-12)


def test_kendall_tau_b_with_ties_matches_scipy():
    rng = np.random.default_rng(1)
    for _ in range(50):
        x = rng.integers(0, 5, size=24).astype(float)
        Y = rng.integers(0, 5, size=(5, 24)).astype(float)
        np.testing.assert_allclose(kendall_tau_b(x, Y), [kendalltau(x, y).statistic for y in Y], atol=1e-12)


def test_kendall_tau_tie_tolerance():
    x = np.array([0.0, 0.05, 2.0, 2.02, 4.0])
    y = np.array([0.03, 0.0, 2.04, 2.0, 4.0])   # same order up to sub-tolerance jitter
    assert kendall_tau_b(x, y)[0] < 1.0
    assert kendall_tau_b(x, y, tie_tolerance=0.5)[0] == pytest.approx(1.0)


@pytest.mark.parametrize("tol", [0.0, 0.5])
def test_rank_numba_scan_matches_numpy_tau(tol):
    from earworm.matchers.rank import _scan_windows, sample_points
    rng = np.random.default_rng(2)
    sv = np.round(rng.normal(60, 3, size=400) * 2) / 2   # half-semitone grid -> many ties
    qs = np.round(rng.normal(60, 3, size=24) * 2) / 2
    offs = sample_points(37, 24)
    starts = np.arange(0, 400 - 37, 3)
    ref = kendall_tau_b(qs, sv[starts[:, None] + offs[None, :]], tie_tolerance=tol)
    np.testing.assert_allclose(_scan_windows(qs, sv, starts, offs, tol), ref, atol=1e-12)


# ---- dtw ----
def test_dtw_exact_subsequence():
    s = np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1.0])
    q = s[3:7].copy()
    cost, st, en = _subseq_dtw(q, s, 2.0)
    assert cost == 0 and (st, en) == (3, 6)


def test_dtw_tempo_bounds():
    s = np.repeat(np.array([0, 3, 0, 5, 1.0]), 2)    # song at half speed
    q = np.array([0, 3, 0, 5, 1.0])
    assert _subseq_dtw(q, s, 2.0)[0] == 0           # 2x compression allowed
    q2 = np.repeat(q, 2)
    s2 = q.copy()
    assert _subseq_dtw(q2, s2, 2.0)[0] == 0          # 2x expansion allowed
    q3 = np.repeat(q, 3)
    assert _subseq_dtw(q3, s2, 2.0)[0] > 0           # 3x is outside the step pattern


def test_dtw_cost_is_clipped():
    q = np.array([0.0, 0.0, 0.0])
    s = np.array([0.0, 10.0, 0.0, 0.0])
    cost, _, _ = _subseq_dtw(q, s, 2.0)
    assert cost == 0.0
    cost, _, _ = _subseq_dtw(np.array([10.0]), np.array([0.0]), 2.0)
    assert cost == 2.0


def _feat(cfg, p10):
    return features_from_pitch(np.asarray(p10, np.float32), cfg)


def test_dtw_matcher_transposition_and_offset(cfg):
    rng = np.random.default_rng(3)
    notes = rng.integers(55, 70, size=40).astype(float)
    song = np.repeat(notes, 30)                  # 300 ms notes
    query = np.repeat(notes[10:22] + 5, 30)       # transposed excerpt starting at 3.0 s
    r = DTWMatcher(cfg).score(_feat(cfg, query), _feat(cfg, song))
    assert r.score == pytest.approx(0.0, abs=1e-6)
    assert r.offset_s == pytest.approx(3.0, abs=0.06)


# ---- fusion ----
def test_zscore_handles_inf_and_constant():
    z = zscore(np.array([1.0, 2.0, -np.inf]))
    assert np.isfinite(z).all() and z[1] > z[0] >= z[2]
    np.testing.assert_allclose(zscore(np.array([3.0, 3.0])), 0)


def test_fuse_with_prior():
    f = fuse({"a": np.array([1.0, 0.0]), "b": np.array([1.0, 0.0])}, {"a": 1, "b": 1}, prior=np.array([0.0, 5.0]))
    assert list(f.order) == [1, 0]
