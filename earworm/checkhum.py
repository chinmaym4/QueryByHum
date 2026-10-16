"""Advisory preflight for hum recordings (`earworm check-hum`).

Measures what `search` will actually see (same decode, silence trim and pitch
pipeline) and flags recordings likely to fail for reasons that are not the
matchers' fault. Warnings never reject anything; thresholds live in
config.toml [checkhum].
"""

from __future__ import annotations

import numpy as np

from .audio import decode, trim_silence
from .config import Section
from .features import features_from_audio


def hum_report(path: str, cfg: Section) -> dict:
    """Measurements and warnings for one hum file. Raises AudioDecodeError if unreadable."""
    sr = int(cfg.audio.sample_rate)
    raw = decode(path, sample_rate=sr, channels=1)          # not peak-normalized: level is what was recorded
    duration = len(raw) / sr
    trimmed = trim_silence(raw, float(cfg.audio.query_trim_top_db))
    rms = float(np.sqrt(np.mean(trimmed.astype(np.float64) ** 2))) if trimmed.size else 0.0
    rms_db = 20 * np.log10(rms) if rms > 0 else float("-inf")
    clipped = float(np.mean(np.abs(raw) >= 0.999))

    f = features_from_audio(raw, cfg, is_query=True)
    voiced = f.pitch_10ms[np.isfinite(f.pitch_10ms)]
    hop = float(cfg.pitch.hop_s)
    voiced_s = len(voiced) * hop
    voiced_frac = len(voiced) / len(f.pitch_10ms) if len(f.pitch_10ms) else 0.0
    rep = {
        "path": str(path), "duration_s": duration, "trimmed_s": len(trimmed) / sr,
        "rms_dbfs": float(rms_db), "clipped_fraction": clipped,
        "voiced_s": voiced_s, "voiced_fraction": voiced_frac,
        "n_notes": int(len(f.notes)),
        "median_note_s": float(np.median(f.notes[:, 1])) if len(f.notes) else float("nan"),
        "pitch_range_st": float(np.percentile(voiced, 95) - np.percentile(voiced, 5)) if voiced.size else 0.0,
    }
    c = cfg.checkhum
    w = []
    if duration < c.min_duration_s:
        w.append(f"short ({duration:.1f} s < {c.min_duration_s:g} s)")
    if duration > c.max_duration_s:
        w.append(f"long ({duration:.1f} s > {c.max_duration_s:g} s)")
    if rms_db < c.min_rms_dbfs:
        w.append(f"quiet ({rms_db:.0f} dBFS < {c.min_rms_dbfs:g})")
    if clipped > c.max_clipped_fraction:
        w.append(f"clipped ({100 * clipped:.2f}% of samples at full scale)")
    if voiced_frac < c.min_voiced_fraction:
        w.append(f"mostly unvoiced ({100 * voiced_frac:.0f}% < {100 * c.min_voiced_fraction:.0f}%)")
    if voiced_s < c.min_voiced_s:
        w.append(f"little voiced audio ({voiced_s:.1f} s < {c.min_voiced_s:g} s)")
    if rep["n_notes"] < c.min_notes:
        w.append(f"few notes ({rep['n_notes']} < {c.min_notes})")
    if rep["pitch_range_st"] < c.min_pitch_range_st:
        w.append(f"flat pitch (range {rep['pitch_range_st']:.1f} st < {c.min_pitch_range_st:g}; not a melody?)")
    rep["warnings"] = w
    return rep


def format_report(rep: dict) -> str:
    med = "n/a" if not np.isfinite(rep["median_note_s"]) else f"{rep['median_note_s']:.2f} s"
    lines = [
        rep["path"],
        f"  duration {rep['duration_s']:.1f} s ({rep['trimmed_s']:.1f} s after trimming silence)",
        f"  level    {rep['rms_dbfs']:.1f} dBFS RMS, {100 * rep['clipped_fraction']:.2f}% clipped",
        f"  voiced   {rep['voiced_s']:.1f} s ({100 * rep['voiced_fraction']:.0f}% of the trimmed hum)",
        f"  notes    {rep['n_notes']}, median length {med}, pitch range {rep['pitch_range_st']:.1f} semitones (p5-p95)",
        "  " + ("OK" if not rep["warnings"] else "WARN: " + "; ".join(rep["warnings"])),
    ]
    return "\n".join(lines)
