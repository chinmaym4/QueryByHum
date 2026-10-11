"""Real-hum evaluation harness (driven by eval/run_eval.py and `earworm eval`)."""

from __future__ import annotations

import csv
import itertools
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import Section
from .search import MATCHER_NAMES, Entry, search

log = logging.getLogger(__name__)

SYSTEMS = (*MATCHER_NAMES, "fused")


@dataclass
class HumLabel:
    hum_file: str
    title: str
    artist: str
    part: str
    in_preview: str
    notes: str = ""


@dataclass
class HumResult:
    label: HumLabel
    song_id: str
    ranks: dict[str, int]
    z: dict[str, np.ndarray] = field(repr=False)   # per-matcher z over library order
    truth_index: int = -1
    match_note: str = ""                           # how the label was mapped to a library song


@dataclass
class Excluded:
    label: HumLabel
    reason: str


def read_labels(path: str | Path) -> list[HumLabel]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        out.append(HumLabel(hum_file=r["hum_file"].strip(), title=r["title"].strip(), artist=r["artist"].strip(),
                            part=(r.get("part") or "other").strip() or "other",
                            in_preview=(r.get("in_preview") or "unknown").strip() or "unknown",
                            notes=(r.get("notes") or "").strip()))
    return out


def label_scores(label: HumLabel, entries: list[Entry]) -> np.ndarray:
    """Fuzzy title+artist score (0..100) of a label row against every library song."""
    from rapidfuzz import fuzz

    return np.array([(fuzz.token_set_ratio(label.title.lower(), e.title.lower())
                      + fuzz.token_set_ratio(label.artist.lower(), e.artist.lower())) / 2 for e in entries])


def find_song(label: HumLabel, entries: list[Entry], threshold: float = 85.0,
              ambiguity_margin: float = 5.0) -> tuple[int | None, str]:
    """Map a label to a library song. Returns (index or None, note). The note names
    the matched song and score, and warns when another song scores within
    `ambiguity_margin` (e.g. a remix or live version that could be the real target)."""
    if not entries:
        return None, "library empty"
    sc = label_scores(label, entries)
    order = np.argsort(-sc, kind="stable")
    i = int(order[0])
    if sc[i] < threshold:
        return None, f"best library match {entries[i].title} - {entries[i].artist} scored {sc[i]:.0f} < {threshold:.0f}"
    note = f"{entries[i].title} - {entries[i].artist} ({sc[i]:.0f})"
    if len(order) > 1 and sc[order[1]] >= sc[i] - ambiguity_margin:
        j = int(order[1])
        note += f"  AMBIGUOUS: also {entries[j].title} - {entries[j].artist} ({sc[j]:.0f})"
    return i, note


def pessimistic_rank(scores: np.ndarray, truth: int) -> int:
    """1 + number of other songs scoring >= the true song: ties count against it,
    and a true song with no possible match (-inf) ranks last."""
    s = np.where(np.isfinite(scores), scores, -np.inf)
    return int(np.sum(s >= s[truth]))   # counts the true song itself, i.e. 1 + others


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def metrics(ranks: list[int]) -> dict[str, float]:
    r = np.asarray(ranks, dtype=float)
    if r.size == 0:
        return {"n": 0, "top1": np.nan, "top3": np.nan, "top5": np.nan, "mrr": np.nan,
                "top5_lo": np.nan, "top5_hi": np.nan}
    lo, hi = wilson_interval(int(np.sum(r <= 5)), int(r.size))
    return {"n": int(r.size), "top1": float(np.mean(r <= 1)), "top3": float(np.mean(r <= 3)),
            "top5": float(np.mean(r <= 5)), "mrr": float(np.mean(1.0 / r)), "top5_lo": lo, "top5_hi": hi}


def run_hums(labels: list[HumLabel], hums_dir: Path, entries: list[Entry], cfg: Section):
    from .audio import decode_mono16k
    from .features import features_from_audio

    ev = cfg.get("eval", {})
    threshold = float(ev.get("label_match_threshold", 85.0))
    margin = float(ev.get("label_ambiguity_margin", 5.0))
    weights = dict(cfg.fusion.weights)
    results: list[HumResult] = []
    excluded: list[Excluded] = []
    for lab in labels:
        ti, note = find_song(lab, entries, threshold, margin)
        if ti is None:
            excluded.append(Excluded(lab, f"song not in library ({note})"))
            continue
        path = hums_dir / lab.hum_file
        try:
            q = features_from_audio(decode_mono16k(path, int(cfg.audio.sample_rate)), cfg, is_query=True)
        except Exception as e:
            log.error("could not process hum %s: %s", path, e)
            excluded.append(Excluded(lab, f"hum unreadable ({type(e).__name__}: {e})"))
            continue
        out = search(q, entries, cfg)
        idx = {r.entry.song_id: r for r in out.results}
        raw = {m: np.array([idx[e.song_id].raw[m] for e in entries]) for m in MATCHER_NAMES}
        z = {m: np.array([idx[e.song_id].z[m] for e in entries]) for m in MATCHER_NAMES}
        res = HumResult(lab, entries[ti].song_id, {}, z, ti, note)
        res.ranks = {m: pessimistic_rank(raw[m], ti) for m in MATCHER_NAMES}
        res.ranks["fused"] = fused_rank(res, weights)
        results.append(res)
    return results, excluded


def fused_rank(res: HumResult, weights: dict[str, float]) -> int:
    f = sum(float(weights.get(m, 0.0)) * res.z[m] for m in MATCHER_NAMES)
    return pessimistic_rank(f, res.truth_index)


def leave_one_out(results: list[HumResult], weights: dict[str, float]):
    """Fused accuracy with all matchers and with each matcher dropped (weight 0).

    Exact, because each matcher's z-scores are computed independently. Returns
    (table, paired) where paired[name] compares per-hum ranks against all-three."""
    full = [fused_rank(r, weights) for r in results]
    table = {"all three": metrics(full)}
    paired = {}
    for drop in MATCHER_NAMES:
        keep = [m for m in MATCHER_NAMES if m != drop]
        name = "+".join(keep)
        w = {m: (float(weights.get(m, 1.0)) if m != drop else 0.0) for m in MATCHER_NAMES}
        ranks = [fused_rank(r, w) for r in results]
        table[name] = metrics(ranks)
        paired[name] = {
            "dropped": drop,
            "better": sum(b < a for a, b in zip(full, ranks)),
            "worse": sum(b > a for a, b in zip(full, ranks)),
            "into_top5": sum(a > 5 >= b for a, b in zip(full, ranks)),
            "out_of_top5": sum(a <= 5 < b for a, b in zip(full, ranks)),
        }
    return table, paired


def weights_search(results: list[HumResult], step: float = 0.1):
    grid = np.round(np.arange(0, 1 + 1e-9, step), 3)
    best = None
    for w in itertools.product(grid, repeat=len(MATCHER_NAMES)):
        if sum(w) == 0:
            continue
        wd = dict(zip(MATCHER_NAMES, w))
        m = metrics([fused_rank(r, wd) for r in results])
        key = (m["top5"], m["mrr"])
        if best is None or key > best[0]:
            best = (key, wd, m)
    return best[1], best[2]


def breakdown(results: list[HumResult], attr: str) -> dict[str, dict[str, dict[str, float]]]:
    groups: dict[str, list[HumResult]] = {}
    for r in results:
        groups.setdefault(getattr(r.label, attr), []).append(r)
    return {g: {s: metrics([r.ranks[s] for r in rs]) for s in SYSTEMS} for g, rs in sorted(groups.items())}


def format_metrics_table(title: str, table: dict[str, dict[str, float]]) -> str:
    lines = [title, f"  {'system':16s} {'n':>4s} {'top1':>6s} {'top3':>6s} {'top5':>6s} {'(95% CI)':>13s} {'mrr':>6s}"]
    for s, m in table.items():
        ci = f"({m['top5_lo']:.2f}-{m['top5_hi']:.2f})"
        lines.append(f"  {s:16s} {m['n']:>4d} {m['top1']:>6.2f} {m['top3']:>6.2f} {m['top5']:>6.2f} {ci:>13s} {m['mrr']:>6.3f}")
    return "\n".join(lines)


def run_eval(cfg: Section, entries: list[Entry], labels_path: Path, hums_dir: Path, results_dir: Path,
             do_weights_search: bool = False) -> str:
    labels = read_labels(labels_path)
    results, excluded = run_hums(labels, hums_dir, entries, cfg)
    out: list[str] = []
    for ex in excluded:
        out.append(f"EXCLUDED: {ex.label.hum_file} ({ex.label.title} - {ex.label.artist}): {ex.reason}")
    if not results:
        out.append("No evaluable hums.")
        return "\n".join(out)
    for r in results:
        if "AMBIGUOUS" in r.match_note:
            out.append(f"WARNING: {r.label.hum_file} label matched {r.match_note}")

    n_lib = sum(e.reason.startswith("song not in library") for e in excluded)
    n_bad = len(excluded) - n_lib
    overall = {s: metrics([r.ranks[s] for r in results]) for s in SYSTEMS}
    out.append(format_metrics_table(
        f"Overall (n={len(results)} evaluated; excluded {n_lib} not in library, {n_bad} unreadable; "
        f"library={len(entries)} songs; ties count against the true song)", overall))
    rows = [("overall", "all", s, m) for s, m in overall.items()]

    weights = dict(cfg.fusion.weights)
    loo, paired = leave_one_out(results, weights)
    out.append(format_metrics_table("Leave-one-out: fused accuracy with each matcher dropped", loo))
    out.append("  vs all three (per hum):  rank better / worse,  entered / left top-5")
    for name, p in paired.items():
        out.append(f"  without {p['dropped']:8s} {p['better']:>4d} / {p['worse']:<4d}        "
                   f"{p['into_top5']:>4d} / {p['out_of_top5']:<4d}")
    rows += [("leave_one_out", "all", s, m) for s, m in loo.items()]
    for attr in ("part", "in_preview"):
        for g, table in breakdown(results, attr).items():
            out.append(format_metrics_table(f"{attr} = {g}", table))
            rows += [(attr, g, s, m) for s, m in table.items()]

    out.append("Per-hum rank of the correct song")
    out.append(f"  {'hum_file':24s} {'part':18s} {'prev':5s} " + " ".join(f"{s:>7s}" for s in SYSTEMS)
               + "  matched song (label score)")
    for r in results:
        out.append(f"  {r.label.hum_file:24s} {r.label.part:18s} {r.label.in_preview:5s} "
                   + " ".join(f"{r.ranks[s]:>7d}" for s in SYSTEMS) + f"  {r.match_note}")

    if do_weights_search:
        wd, m = weights_search(results)
        out.append("WARNING: weights grid search on a small eval set overfits; treat the result as a hint, "
                   "not a tuned value.")
        out.append(f"Best weights {wd}: top1={m['top1']:.2f} top3={m['top3']:.2f} "
                   f"top5={m['top5']:.2f} mrr={m['mrr']:.3f} (current: {dict(cfg.fusion.weights)})")

    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    per_hum = results_dir / f"{ts}.csv"
    with open(per_hum, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hum_file", "title", "artist", "part", "in_preview", "song_id", *[f"rank_{s}" for s in SYSTEMS],
                    "label_match"])
        for r in results:
            w.writerow([r.label.hum_file, r.label.title, r.label.artist, r.label.part, r.label.in_preview,
                        r.song_id, *[r.ranks[s] for s in SYSTEMS], r.match_note])
        for ex in excluded:
            lab = ex.label
            w.writerow([lab.hum_file, lab.title, lab.artist, lab.part, lab.in_preview, "EXCLUDED",
                        *[""] * len(SYSTEMS), ex.reason])
    summary = results_dir / f"{ts}_metrics.csv"
    with open(summary, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group_by", "group", "system", "n", "top1", "top3", "top5", "top5_ci95_lo", "top5_ci95_hi", "mrr"])
        for g_by, g, s, m in rows:
            w.writerow([g_by, g, s, m["n"], m["top1"], m["top3"], m["top5"], m["top5_lo"], m["top5_hi"], m["mrr"]])
    out.append(f"Saved {per_hum} and {summary}")
    return "\n".join(out)
