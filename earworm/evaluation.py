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


def find_song(label: HumLabel, entries: list[Entry], threshold: float = 85.0) -> int | None:
    from rapidfuzz import fuzz

    best, best_i = 0.0, None
    for i, e in enumerate(entries):
        s = (fuzz.token_set_ratio(label.title.lower(), e.title.lower())
             + fuzz.token_set_ratio(label.artist.lower(), e.artist.lower())) / 2
        if s > best:
            best, best_i = s, i
    return best_i if best >= threshold else None


def metrics(ranks: list[int]) -> dict[str, float]:
    r = np.asarray(ranks, dtype=float)
    if r.size == 0:
        return {"n": 0, "top1": np.nan, "top3": np.nan, "top5": np.nan, "mrr": np.nan}
    return {"n": int(r.size), "top1": float(np.mean(r <= 1)), "top3": float(np.mean(r <= 3)),
            "top5": float(np.mean(r <= 5)), "mrr": float(np.mean(1.0 / r))}


def run_hums(labels: list[HumLabel], hums_dir: Path, entries: list[Entry], cfg: Section):
    from .audio import decode_mono16k
    from .features import features_from_audio

    results: list[HumResult] = []
    missing: list[HumLabel] = []
    for lab in labels:
        ti = find_song(lab, entries)
        if ti is None:
            missing.append(lab)
            continue
        path = hums_dir / lab.hum_file
        try:
            q = features_from_audio(decode_mono16k(path, int(cfg.audio.sample_rate)), cfg, is_query=True)
        except Exception as e:
            log.error("could not process hum %s: %s", path, e)
            missing.append(lab)
            continue
        out = search(q, entries, cfg)
        ranks = {m: int(np.flatnonzero(out.matcher_order[m] == ti)[0]) + 1 for m in MATCHER_NAMES}
        ranks["fused"] = next(r.rank for r in out.results if r.entry is entries[ti])
        idx = {r.entry.song_id: r for r in out.results}
        z = {m: np.array([idx[e.song_id].z[m] for e in entries]) for m in MATCHER_NAMES}
        results.append(HumResult(lab, entries[ti].song_id, ranks, z, ti))
    return results, missing


def fused_rank(res: HumResult, weights: dict[str, float]) -> int:
    f = sum(weights[m] * res.z[m] for m in MATCHER_NAMES)
    return int(np.sum(f > f[res.truth_index])) + 1


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
    lines = [title, f"  {'system':10s} {'n':>4s} {'top1':>6s} {'top3':>6s} {'top5':>6s} {'mrr':>6s}"]
    for s, m in table.items():
        lines.append(f"  {s:10s} {m['n']:>4d} {m['top1']:>6.2f} {m['top3']:>6.2f} {m['top5']:>6.2f} {m['mrr']:>6.3f}")
    return "\n".join(lines)


def run_eval(cfg: Section, entries: list[Entry], labels_path: Path, hums_dir: Path, results_dir: Path,
             do_weights_search: bool = False) -> str:
    labels = read_labels(labels_path)
    results, missing = run_hums(labels, hums_dir, entries, cfg)
    out: list[str] = []
    for lab in missing:
        out.append(f"EXCLUDED: {lab.hum_file} ({lab.title} - {lab.artist}): song not in library or hum unreadable")
    if not results:
        out.append("No evaluable hums.")
        return "\n".join(out)

    overall = {s: metrics([r.ranks[s] for r in results]) for s in SYSTEMS}
    out.append(format_metrics_table(f"Overall (n={len(results)}, library={len(entries)} songs)", overall))
    rows = [("overall", "all", s, m) for s, m in overall.items()]
    for attr in ("part", "in_preview"):
        for g, table in breakdown(results, attr).items():
            out.append(format_metrics_table(f"{attr} = {g}", table))
            rows += [(attr, g, s, m) for s, m in table.items()]

    out.append("Per-hum rank of the correct song")
    out.append(f"  {'hum_file':24s} {'part':18s} {'prev':5s} " + " ".join(f"{s:>7s}" for s in SYSTEMS) + "  song")
    for r in results:
        out.append(f"  {r.label.hum_file:24s} {r.label.part:18s} {r.label.in_preview:5s} "
                   + " ".join(f"{r.ranks[s]:>7d}" for s in SYSTEMS) + f"  {r.song_id}")

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
        w.writerow(["hum_file", "title", "artist", "part", "in_preview", "song_id", *[f"rank_{s}" for s in SYSTEMS]])
        for r in results:
            w.writerow([r.label.hum_file, r.label.title, r.label.artist, r.label.part, r.label.in_preview,
                        r.song_id, *[r.ranks[s] for s in SYSTEMS]])
        for lab in missing:
            w.writerow([lab.hum_file, lab.title, lab.artist, lab.part, lab.in_preview, "EXCLUDED", *[""] * len(SYSTEMS)])
    summary = results_dir / f"{ts}_metrics.csv"
    with open(summary, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group_by", "group", "system", "n", "top1", "top3", "top5", "mrr"])
        for g_by, g, s, m in rows:
            w.writerow([g_by, g, s, m["n"], m["top1"], m["top3"], m["top5"], m["mrr"]])
    out.append(f"Saved {per_hum} and {summary}")
    return "\n".join(out)
