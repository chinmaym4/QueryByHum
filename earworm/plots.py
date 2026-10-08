"""`earworm explain`: query contour overlaid on the best-matching song segment, per matcher."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import Section
from .features import Features
from .matchers.base import MatchResult


def _t(n: int, hop: float) -> np.ndarray:
    return np.arange(n) * hop


def _query_shift(query: Features, song: Features, start_s: float, end_s: float) -> float:
    """Median-align the query to a song segment (for display only)."""
    qv = query.pitch_50ms[np.isfinite(query.pitch_50ms)]
    a, b = int(start_s / song.hop_50ms), int(np.ceil(end_s / song.hop_50ms))
    seg = song.pitch_50ms[a:b]
    seg = seg[np.isfinite(seg)]
    if qv.size == 0 or seg.size == 0:
        return 0.0
    return float(np.median(seg) - np.median(qv))


def explain_plot(query: Features, song: Features, results: dict[str, MatchResult], title: str,
                 out_path: str | Path, cfg: Section, z: dict[str, float] | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .matchers.dtw import DTWMatcher

    names = list(results)
    fig, axes = plt.subplots(len(names), 1, figsize=(12, 3.2 * len(names)), squeeze=False)
    q_dur = len(query.pitch_10ms) * query.hop_s
    for ax, name in zip(axes[:, 0], names):
        r = results[name]
        start = r.offset_s
        end = r.detail.get("end_s", start + q_dur)
        pad = 1.0
        lo, hi = max(0.0, start - pad), end + pad
        st = _t(len(song.pitch_10ms), song.hop_s)
        m = (st >= lo) & (st <= hi)
        ax.plot(st[m], song.pitch_10ms[m], color="0.55", lw=1.2, label="song (10 ms)")
        if name == "dtw" and np.isfinite(r.score):
            qt, s_t, qp, _ = DTWMatcher(cfg).alignment(query, song, r)
            ax.plot(s_t, qp, ".", color="C0", ms=4, label="query, DTW-warped onto song time")
        else:
            span = max(end - start, 1e-3)
            scale = span / max(q_dur, 1e-3)
            shift = _query_shift(query, song, start, end)
            qt = _t(len(query.pitch_10ms), query.hop_s) * scale + start
            ax.plot(qt, query.pitch_10ms + shift, color="C0", lw=1.2,
                    label=f"query (time x{scale:.2f}, shifted {shift:+.1f} st)")
        if name == "contour":
            a, b = r.detail.get("start_note", 0), r.detail.get("end_note", -1)
            for on, du, p in song.notes[a : b + 1]:
                ax.hlines(p, on, on + du, color="C3", lw=3, alpha=0.6)
            ax.plot([], [], color="C3", lw=3, alpha=0.6, label="song notes in aligned region")
        ax.axvspan(start, end, color="C2", alpha=0.08)
        ztxt = f", z={z[name]:+.2f}" if z and name in z else ""
        ax.set_title(f"{name}: score={r.score:.3f}{ztxt}, offset={start:.2f}s")
        ax.set_xlabel("song time (s)")
        ax.set_ylabel("MIDI")
        ax.set_xlim(lo, hi)
        ax.legend(loc="best", fontsize=8, framealpha=0.85)
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
