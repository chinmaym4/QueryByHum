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
    true_offset_s: float | None = None   # where the hummed part starts, in seconds into the INDEXED audio


@dataclass
class HumResult:
    label: HumLabel
    song_id: str
    ranks: dict[str, int]
    z: dict[str, np.ndarray] = field(repr=False)   # per-matcher z over library order
    truth_index: int = -1
    match_note: str = ""                           # how the label was mapped to a library song
    offsets: dict[str, float] = field(default_factory=dict)   # dtw/contour/rank/fused offset of the TRUE song (NaN = none)
    offset_matcher: str = ""                       # matcher the fused offset came from
    spread_s: float = float("nan")                 # max - min over the three matchers' offsets
    duration_s: float | None = None                # length of the indexed audio, if known


@dataclass
class Excluded:
    label: HumLabel
    reason: str


def _parse_offset(raw: str, hum_file: str) -> float | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        t = float(raw)
    except ValueError:
        raise ValueError(f"labels.csv: {hum_file}: true_offset_s {raw!r} is not a number of seconds") from None
    if not np.isfinite(t) or t < 0:
        raise ValueError(f"labels.csv: {hum_file}: true_offset_s must be >= 0, got {raw!r}")
    return t


def read_labels(path: str | Path) -> list[HumLabel]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        hum = r["hum_file"].strip()
        out.append(HumLabel(hum_file=hum, title=r["title"].strip(), artist=r["artist"].strip(),
                            part=(r.get("part") or "other").strip() or "other",
                            in_preview=(r.get("in_preview") or "unknown").strip() or "unknown",
                            notes=(r.get("notes") or "").strip(),
                            true_offset_s=_parse_offset(r.get("true_offset_s"), hum)))
    return out


def matcher_offsets(result) -> tuple[dict[str, float], float]:
    """Offsets (seconds) each matcher reported for one song, and their max spread.

    A matcher with no usable match (non-finite score) contributes NaN. The spread
    is max - min over the matchers that did report, NaN if fewer than two did."""
    off = {}
    for m in MATCHER_NAMES:
        r = result.per_matcher.get(m)
        off[m] = float(r.offset_s) if r is not None and np.isfinite(r.score) else float("nan")
    vals = [v for v in off.values() if np.isfinite(v)]
    spread = float(max(vals) - min(vals)) if len(vals) >= 2 else float("nan")
    return off, spread


def spread_by_group(results: list["HumResult"], tolerance_s: float) -> dict[str, dict[str, float]]:
    """Matcher offset spread (max - min over dtw/contour/rank) split by whether the
    fused ranking put the correct song first. Per group: hums in the group, hums
    with a spread (needs >= 2 matchers with an offset), median, max, and how many
    spreads exceed the tolerance."""
    groups = {"fused ranked song 1st": [r for r in results if r.ranks["fused"] == 1],
              "all other hums": [r for r in results if r.ranks["fused"] != 1]}
    out = {}
    for name, rs in groups.items():
        sp = np.array([r.spread_s for r in rs if np.isfinite(r.spread_s)], dtype=float)
        out[name] = {"hums": len(rs), "with_spread": int(sp.size),
                     "median": float(np.median(sp)) if sp.size else float("nan"),
                     "max": float(sp.max()) if sp.size else float("nan"),
                     "over_tol": int(np.sum(sp > tolerance_s))}
    return out


def offset_accuracy(results: list["HumResult"], tolerance_s: float, only_fused_top1: bool = False):
    """Per system: n, fraction within tolerance of true_offset_s, median |error|.

    Uses every hum that has true_offset_s (optionally only those where the fused
    ranking put the true song first). Offsets are those of the TRUE song."""
    rows = [r for r in results if r.label.true_offset_s is not None
            and (not only_fused_top1 or r.ranks["fused"] == 1)]
    table = {}
    for sysname in SYSTEMS:
        errs = np.array([abs(r.offsets[sysname] - r.label.true_offset_s) for r in rows], dtype=float)
        ok = np.isfinite(errs)
        # a hum whose matcher gave no offset counts as a miss, not as missing data
        table[sysname] = {"n": len(errs), "n_no_offset": int((~ok).sum()),
                          "within": float(np.sum(errs[ok] <= tolerance_s) / len(errs)) if len(errs) else float("nan"),
                          "median_abs_err": float(np.median(errs[ok])) if ok.any() else float("nan")}
    return table


def sample_playlist(rows: list, n: int, seed: int) -> list[tuple[int, object]]:
    """Seeded random sample of playlist rows, as (1-based playlist row, row) sorted by row.

    Deterministic for a given (rows, n, seed). Nothing about the rows (ingest
    success, difficulty, artist) influences which are drawn."""
    import random

    if n > len(rows):
        raise ValueError(f"cannot sample {n} songs from a playlist of {len(rows)}")
    picked = random.Random(seed).sample(range(len(rows)), n)
    return [(i + 1, rows[i]) for i in sorted(picked)]


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
        true_res = idx[entries[ti].song_id]
        res.offsets, res.spread_s = matcher_offsets(true_res)
        res.offsets["fused"] = float(true_res.offset_s)
        res.offset_matcher = true_res.offset_matcher
        res.duration_s = entries[ti].meta.get("duration_s") or None
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

    tol = float(cfg.get("eval", {}).get("offset_tolerance_s", 1.0))
    off_rows = []
    labeled = [r for r in results if r.label.true_offset_s is not None]
    for r in labeled:
        if r.label.in_preview == "n":
            out.append(f"WARNING: {r.label.hum_file} has true_offset_s but in_preview=n; the hummed part is "
                       f"outside the indexed audio, so its offset cannot be matched")
        if r.duration_s and r.label.true_offset_s > r.duration_s:
            out.append(f"WARNING: {r.label.hum_file} true_offset_s={r.label.true_offset_s:.1f} is beyond the "
                       f"indexed audio ({r.duration_s:.1f} s); true_offset_s must be seconds into the indexed "
                       f"audio (the 30 s preview for Deezer songs), not into the full track")
    if labeled:
        out.append(f"Offset accuracy (seconds into the indexed audio; within +/-{tol:g} s; "
                   f"{len(labeled)} of {len(results)} hums have true_offset_s)")
        for title, only in (("  all hums with true_offset_s", False), ("  only hums where fused ranked the song 1st", True)):
            table = offset_accuracy(results, tol, only_fused_top1=only)
            out.append(f"{title}  (n={next(iter(table.values()))['n']})")
            out.append(f"    {'system':10s} {'within':>7s} {'median |err| (s)':>17s} {'no offset':>10s}")
            for sysname, m in table.items():
                out.append(f"    {sysname:10s} {m['within']:>7.2f} {m['median_abs_err']:>17.2f} {m['n_no_offset']:>10d}")
            off_rows += [("all" if not only else "fused_top1", sysname, m) for sysname, m in table.items()]
        out.append("  (fused offset comes from DTW unless DTW had no match)")
    else:
        out.append("Offset accuracy: no hum has true_offset_s (optional labels.csv column)")

    out.append("Per-hum offsets in seconds into the indexed audio; spread = max - min over dtw/contour/rank")
    out.append(f"  {'hum_file':24s} {'true':>6s} {'dtw':>7s} {'contour':>8s} {'rank':>7s} {'fused':>7s} "
               f"{'(from)':>8s} {'spread':>7s} {'fused err':>10s}")
    fmt = lambda v: "     - " if not np.isfinite(v) else f"{v:7.2f}"
    for r in results:
        true = r.label.true_offset_s
        err = abs(r.offsets["fused"] - true) if true is not None else float("nan")
        out.append(f"  {r.label.hum_file:24s} {'-' if true is None else f'{true:6.2f}':>6s} {fmt(r.offsets['dtw'])} "
                   f"{fmt(r.offsets['contour']):>8s} {fmt(r.offsets['rank'])} {fmt(r.offsets['fused'])} "
                   f"{r.offset_matcher:>8s} {fmt(r.spread_s)} {'-' if not np.isfinite(err) else f'{err:.2f}':>10s}")
    spreads = [(r.spread_s, r.label.hum_file) for r in results if np.isfinite(r.spread_s)]
    if spreads:
        worst = max(spreads)
        wide = sum(sp > tol for sp, _ in spreads)
        out.append(f"  max spread between matchers: {worst[0]:.2f} s ({worst[1]}); "
                   f"{wide} of {len(spreads)} hums have spread > {tol:g} s")
    out.append(f"Matcher offset spread by group (spread = max - min over dtw/contour/rank; tolerance {tol:g} s)")
    out.append(f"  {'group':24s} {'hums':>5s} {'w/ spread':>10s} {'median (s)':>11s} {'max (s)':>8s} {'> tol':>6s}")
    for name, g in spread_by_group(results, tol).items():
        med = "-" if not np.isfinite(g["median"]) else f"{g['median']:.2f}"
        mx = "-" if not np.isfinite(g["max"]) else f"{g['max']:.2f}"
        out.append(f"  {name:24s} {g['hums']:>5d} {g['with_spread']:>10d} {med:>11s} {mx:>8s} {g['over_tol']:>6d}")

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
        off_cols = ["true_offset_s", *[f"offset_{s}" for s in SYSTEMS], "offset_fused_from", "offset_spread_s"]
        w.writerow(["hum_file", "title", "artist", "part", "in_preview", "song_id", *[f"rank_{s}" for s in SYSTEMS],
                    *off_cols, "label_match"])
        blank = lambda v: "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else v
        for r in results:
            w.writerow([r.label.hum_file, r.label.title, r.label.artist, r.label.part, r.label.in_preview,
                        r.song_id, *[r.ranks[s] for s in SYSTEMS],
                        blank(r.label.true_offset_s), *[blank(r.offsets[s]) for s in SYSTEMS], r.offset_matcher,
                        blank(r.spread_s), r.match_note])
        for ex in excluded:
            lab = ex.label
            w.writerow([lab.hum_file, lab.title, lab.artist, lab.part, lab.in_preview, "EXCLUDED",
                        *[""] * len(SYSTEMS), *[""] * len(off_cols), ex.reason])
    summary = results_dir / f"{ts}_metrics.csv"
    with open(summary, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group_by", "group", "system", "n", "top1", "top3", "top5", "top5_ci95_lo", "top5_ci95_hi", "mrr"])
        for g_by, g, s, m in rows:
            w.writerow([g_by, g, s, m["n"], m["top1"], m["top3"], m["top5"], m["top5_lo"], m["top5_hi"], m["mrr"]])
    saved = [per_hum, summary]
    if off_rows:
        offs = results_dir / f"{ts}_offset_accuracy.csv"
        with open(offs, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["subset", "system", "n", "n_no_offset", f"within_{tol:g}s", "median_abs_err_s"])
            for subset, sysname, m in off_rows:
                w.writerow([subset, sysname, m["n"], m["n_no_offset"], m["within"], m["median_abs_err"]])
        saved.append(offs)
    out.append("Saved " + " and ".join(str(p) for p in saved))
    return "\n".join(out)
