"""torchcrepe -> cleaned MIDI pitch curve (shared by songs and queries)."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from .config import Section, resolve_device


def hz_to_midi(hz: np.ndarray) -> np.ndarray:
    hz = np.asarray(hz, dtype=np.float64)
    out = np.full(hz.shape, np.nan)
    ok = np.isfinite(hz) & (hz > 0)
    out[ok] = 69.0 + 12.0 * np.log2(hz[ok] / 440.0)
    return out


def midi_to_hz(midi: np.ndarray) -> np.ndarray:
    return 440.0 * 2.0 ** ((np.asarray(midi, dtype=np.float64) - 69.0) / 12.0)


def voiced_runs(x: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of maximal non-NaN runs."""
    v = np.isfinite(x).astype(np.int8)
    if v.size == 0:
        return []
    d = np.diff(np.concatenate([[0], v, [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def median_filter_runs(midi: np.ndarray, width: int) -> np.ndarray:
    out = midi.copy()
    if width <= 1:
        return out
    for s, e in voiced_runs(midi):
        out[s:e] = median_filter(midi[s:e], size=width, mode="nearest")
    return out


def fix_octave_errors(midi: np.ndarray, jump_min: float, jump_max: float, max_frames: int) -> np.ndarray:
    """Fold back short excursions that jump by ~an octave and then return.

    Within a voiced run, a jump of jump_min..jump_max semitones followed, within
    fewer than `max_frames` frames, by a jump of the opposite sign in the same
    range is treated as an octave error and the excursion is shifted by 12.
    """
    out = midi.copy()
    for s, e in voiced_runs(midi):
        seg = out[s:e]
        i = 1
        while i < len(seg):
            d = seg[i] - seg[i - 1]
            if jump_min <= abs(d) <= jump_max:
                sign = np.sign(d)
                returned = False
                for j in range(i + 1, min(i + max_frames, len(seg))):
                    dj = seg[j] - seg[j - 1]
                    if np.sign(dj) == -sign and jump_min <= abs(dj) <= jump_max:
                        seg[i:j] -= sign * 12.0
                        i = j
                        returned = True
                        break
                if not returned:
                    i += 1
            else:
                i += 1
        out[s:e] = seg
    return out


def drop_short_runs(midi: np.ndarray, min_frames: int) -> np.ndarray:
    out = midi.copy()
    for s, e in voiced_runs(midi):
        if e - s < min_frames:
            out[s:e] = np.nan
    return out


def downsample_nanmedian(x: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return x.astype(np.float32)
    n = int(np.ceil(len(x) / factor))
    pad = np.full(n * factor, np.nan)
    pad[: len(x)] = x
    blocks = pad.reshape(n, factor)
    out = np.full(n, np.nan)
    ok = np.isfinite(blocks).any(axis=1)
    if ok.any():
        out[ok] = np.nanmedian(blocks[ok], axis=1)
    return out.astype(np.float32)


def clean_pitch(hz: np.ndarray, periodicity: np.ndarray, cfg: Section) -> np.ndarray:
    """Spec steps 3-7: voicing, Hz->MIDI, median filter, octave fix, drop short runs."""
    p = cfg.pitch
    hz = np.asarray(hz, dtype=np.float64).copy()
    hz[np.asarray(periodicity) < p.periodicity_threshold] = np.nan
    midi = hz_to_midi(hz)
    midi = median_filter_runs(midi, int(p.median_width))
    max_frames = int(round(p.octave_max_s / p.hop_s))
    midi = fix_octave_errors(midi, p.octave_jump_min, p.octave_jump_max, max_frames)
    midi = drop_short_runs(midi, int(round(p.min_voiced_run_s / p.hop_s)))
    return midi.astype(np.float32)


_MODEL_CACHE: dict = {}


def crepe_model_name(cfg: Section, device: str) -> str:
    m = cfg.pitch.model
    if m != "auto":
        return m
    return "tiny" if device == "cpu" else "full"


def run_crepe(audio: np.ndarray, cfg: Section) -> tuple[np.ndarray, np.ndarray]:
    """Return (hz, periodicity) at cfg.pitch.hop_s for 16 kHz mono audio."""
    import torch
    import torchcrepe

    sr = int(cfg.audio.sample_rate)
    device = resolve_device(cfg)
    model = crepe_model_name(cfg, device)
    hop = int(round(cfg.pitch.hop_s * sr))
    x = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None]
    with torch.inference_mode():
        hz, periodicity = torchcrepe.predict(
            x, sr, hop_length=hop, fmin=cfg.pitch.fmin, fmax=cfg.pitch.fmax,
            model=model, batch_size=int(cfg.pitch.batch_size), device=device,
            return_periodicity=True, pad=True,
        )
    return hz[0].cpu().numpy(), periodicity[0].cpu().numpy()


def extract_pitch(audio: np.ndarray, cfg: Section) -> tuple[np.ndarray, np.ndarray]:
    """16 kHz mono audio -> (pitch_10ms, pitch_50ms) in MIDI, NaN where unvoiced."""
    if len(audio) < int(cfg.audio.sample_rate * cfg.pitch.hop_s) * 2:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty
    hz, periodicity = run_crepe(audio, cfg)
    p10 = clean_pitch(hz, periodicity, cfg)
    p50 = downsample_nanmedian(p10, int(cfg.pitch.downsample_factor))
    return p10, p50


def trim_unvoiced_edges(pitch: np.ndarray) -> tuple[np.ndarray, int]:
    """Drop leading/trailing NaN frames; returns (trimmed, n_leading_dropped)."""
    idx = np.flatnonzero(np.isfinite(pitch))
    if idx.size == 0:
        return pitch[:0], 0
    return pitch[idx[0] : idx[-1] + 1], int(idx[0])
