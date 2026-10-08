"""Synthetic accuracy tests (SPEC.md "Tests"). Thresholds are from the spec; do
not loosen them to make a failing run pass -- fix the matcher instead."""

import numpy as np
import pytest

from earworm.features import features_from_pitch
from earworm.pitch import clean_pitch, trim_unvoiced_edges
from earworm.search import MATCHER_NAMES, Entry, search

OFFSET_TOL_S = 1.0


def song_features(crepe, cfg):
    hz, per = crepe
    return features_from_pitch(clean_pitch(hz, per, cfg), cfg)


def query_features(crepe, cfg):
    hz, per = crepe
    p, _ = trim_unvoiced_edges(clean_pitch(hz, per, cfg))
    return features_from_pitch(p, cfg)


def run_all(synthetic_set, cfg):
    entries = [Entry(song_id=f"s{i:02d}", title=f"song {i}", artist="synth",
                     stems={"vocals": song_features(c, cfg)})
               for i, c in enumerate(synthetic_set.song_crepe)]
    rows = []
    for q, qc in zip(synthetic_set.queries, synthetic_set.query_crepe):
        out = search(query_features(qc, cfg), entries, cfg)
        row = {"song": q.song, "true_offset": q.true_offset_s}
        for m in MATCHER_NAMES:
            order = list(out.matcher_order[m])
            row[f"{m}_rank"] = order.index(q.song) + 1
            res = next(r for r in out.results if r.entry.song_id == f"s{q.song:02d}")
            row[f"{m}_offset"] = res.per_matcher[m].offset_s
        fused_ids = [int(r.entry.song_id[1:]) for r in out.results]
        row["fused_rank"] = fused_ids.index(q.song) + 1
        row["fused_offset"] = out.results[row["fused_rank"] - 1].offset_s
        rows.append(row)
    return rows


@pytest.fixture(scope="module")
def results(synthetic_set, cfg):
    rows = run_all(synthetic_set, cfg)
    print("\nsynthetic results (n=%d)" % len(rows))
    for m in (*MATCHER_NAMES, "fused"):
        ranks = np.array([r[f"{m}_rank"] for r in rows])
        top1 = [r for r in rows if r[f"{m}_rank"] == 1]
        off_ok = np.mean([abs(r[f"{m}_offset"] - r["true_offset"]) <= OFFSET_TOL_S for r in top1]) if top1 else 0
        print(f"  {m:8s} top1={np.mean(ranks == 1):.2f} top3={np.mean(ranks <= 3):.2f} "
              f"top5={np.mean(ranks <= 5):.2f} mrr={np.mean(1 / ranks):.3f} offset_ok={off_ok:.2f}")
    return rows


@pytest.mark.slow
@pytest.mark.parametrize("matcher", MATCHER_NAMES)
def test_matcher_top3(results, matcher):
    acc = np.mean([r[f"{matcher}_rank"] <= 3 for r in results])
    assert acc >= 0.80, f"{matcher} top-3 accuracy {acc:.2f} < 0.80"


@pytest.mark.slow
def test_fused_top1(results):
    acc = np.mean([r["fused_rank"] == 1 for r in results])
    assert acc >= 0.85, f"fused top-1 accuracy {acc:.2f} < 0.85"


@pytest.mark.slow
@pytest.mark.parametrize("matcher", (*MATCHER_NAMES, "fused"))
def test_offsets(results, matcher):
    correct = [r for r in results if r[f"{matcher}_rank"] == 1]
    assert correct, f"{matcher} has no correct top-1 matches"
    ok = np.mean([abs(r[f"{matcher}_offset"] - r["true_offset"]) <= OFFSET_TOL_S for r in correct])
    assert ok >= 0.80, f"{matcher} offset within {OFFSET_TOL_S}s for {ok:.2f} of correct matches < 0.80"
