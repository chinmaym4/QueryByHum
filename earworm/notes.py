"""Note segmentation from a 10 ms MIDI pitch curve.

Output: float32[n, 3] rows of (onset_s, duration_s, midi_pitch).
"""

from __future__ import annotations

import numpy as np

from .config import Section


def _segment_frames(pitch: np.ndarray, gap_frames: int, change: float, change_frames: int) -> list[list[int]]:
    """Group voiced frame indices into notes (before short-note merging)."""
    notes: list[list[int]] = []
    cur: list[int] = []
    pending: list[int] = []     # frames deviating from the running median
    pending_sign = 0.0
    last_voiced = -(10**9)

    def close():
        nonlocal cur, pending, pending_sign
        if cur:
            notes.append(cur)
        cur = list(pending)
        pending = []
        pending_sign = 0.0

    for i in range(len(pitch)):
        x = pitch[i]
        if not np.isfinite(x):
            continue
        if cur and i - last_voiced - 1 > gap_frames:
            cur.extend(pending)
            pending = []
            pending_sign = 0.0
            notes.append(cur)
            cur = []
        last_voiced = i
        if not cur:
            cur = [i]
            continue
        med = float(np.median(pitch[cur]))
        dev = x - med
        if abs(dev) > change and (not pending or np.sign(dev) == pending_sign):
            pending.append(i)
            pending_sign = np.sign(dev)
            if len(pending) >= change_frames:
                close()
        else:
            # deviation did not persist: those frames belong to the current note
            cur.extend(pending)
            pending = []
            pending_sign = 0.0
            if abs(dev) > change:
                pending = [i]
                pending_sign = np.sign(dev)
            else:
                cur.append(i)
    cur.extend(pending)
    if cur:
        notes.append(cur)
    return notes


def _merge_short(groups: list[list[int]], pitch: np.ndarray, min_frames: int, gap_frames: int) -> list[list[int]]:
    """Merge notes shorter than min_frames into the adjacent note with the closest pitch.

    Only time-adjacent neighbours (separated by <= gap_frames unvoiced frames)
    are merge candidates; an isolated short note is kept as is.
    """
    groups = [list(g) for g in groups]
    while True:
        meds = [float(np.median(pitch[g])) for g in groups]
        lengths = [g[-1] - g[0] + 1 for g in groups]
        order = sorted((n, k) for k, n in enumerate(lengths) if n < min_frames)
        merged = False
        for _, k in order:
            cands = []
            if k > 0 and groups[k][0] - groups[k - 1][-1] - 1 <= gap_frames:
                cands.append(k - 1)
            if k + 1 < len(groups) and groups[k + 1][0] - groups[k][-1] - 1 <= gap_frames:
                cands.append(k + 1)
            if not cands:
                continue
            tgt = min(cands, key=lambda c: abs(meds[c] - meds[k]))
            a, b = sorted((k, tgt))
            groups[a] = groups[a] + groups[b]
            del groups[b]
            merged = True
            break
        if not merged:
            return groups


def segment_notes(pitch_10ms: np.ndarray, cfg: Section) -> np.ndarray:
    hop = cfg.pitch.hop_s
    n = cfg.notes
    pitch = np.asarray(pitch_10ms, dtype=np.float64)
    gap_frames = int(round(n.gap_s / hop))
    groups = _segment_frames(pitch, gap_frames, n.pitch_change, int(round(n.pitch_change_min_s / hop)))
    groups = _merge_short(groups, pitch, int(round(n.min_note_s / hop)), gap_frames)
    out = np.zeros((len(groups), 3), dtype=np.float32)
    for k, g in enumerate(groups):
        out[k] = (g[0] * hop, (g[-1] - g[0] + 1) * hop, np.median(pitch[g]))
    return out
