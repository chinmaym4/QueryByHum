"""Synthetic melodies, rendering, and query generation for tests.

Melodies are lists of (midi_pitch, duration_s) notes rendered as sine tones with
light harmonics at 16 kHz. Each note is followed by a short articulation rest
(part of its duration) so repeated pitches stay separate notes, like hummed "hm hm".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SR = 16000
ARTICULATION_S = 0.1   # silent tail of every note
FADE_S = 0.01


@dataclass
class Melody:
    pitches: np.ndarray    # MIDI (float)
    durations: np.ndarray  # seconds (inter-onset)

    @property
    def onsets(self) -> np.ndarray:
        return np.concatenate([[0.0], np.cumsum(self.durations)[:-1]])


def random_melody(rng: np.random.Generator, n_min=40, n_max=80) -> Melody:
    n = int(rng.integers(n_min, n_max + 1))
    steps = np.arange(-5, 6)
    w = np.array([1, 2, 4, 8, 10, 6, 10, 8, 4, 2, 1], dtype=float)
    w /= w.sum()
    p = [float(rng.integers(55, 68))]
    for _ in range(n - 1):
        nxt = p[-1] + rng.choice(steps, p=w)
        if nxt < 50 or nxt > 76:   # reflect at the edges of a comfortable range
            nxt = p[-1] - (nxt - p[-1])
        p.append(float(nxt))
    beat = float(rng.uniform(0.4, 0.6))
    beats = rng.choice([0.5, 1.0, 1.5, 2.0], size=n, p=[0.35, 0.4, 0.1, 0.15])
    return Melody(np.array(p), beats * beat)


def render(m: Melody, sr: int = SR, lead_s: float = 0.3, tail_s: float = 0.3,
           rng: np.random.Generator | None = None) -> np.ndarray:
    total = lead_s + float(m.durations.sum()) + tail_s
    out = np.zeros(int(total * sr) + 1, dtype=np.float64)
    fade = int(FADE_S * sr)
    for onset, dur, pitch in zip(m.onsets, m.durations, m.pitches):
        sound = max(dur - ARTICULATION_S, 0.05)
        n = int(sound * sr)
        t = np.arange(n) / sr
        f0 = 440.0 * 2 ** ((pitch - 69) / 12)
        x = np.sin(2 * np.pi * f0 * t) + 0.3 * np.sin(4 * np.pi * f0 * t) + 0.1 * np.sin(6 * np.pi * f0 * t)
        env = np.ones(n)
        k = min(fade, n // 2)
        if k > 0:
            env[:k] = np.linspace(0, 1, k)
            env[-k:] = np.linspace(1, 0, k)
        a = int((lead_s + onset) * sr)
        out[a : a + n] += 0.5 * x * env
    if rng is not None:
        out += rng.normal(0, 0.003, size=out.shape)
    return out.astype(np.float32)


@dataclass
class Query:
    song: int
    start_note: int
    true_offset_s: float       # excerpt start in song time (render lead included)
    melody: Melody


def make_query(rng: np.random.Generator, lib: list[Melody], song: int, lead_s: float = 0.3,
               fractional: bool = False, intonation_sd: float = 0.25) -> Query:
    """SPEC query degradations. With fractional=True, the transposition is a
    continuous offset (e.g. +2.37) and every note also gets a continuous
    intonation error (normal, sd=intonation_sd semitones), as in real humming:
    otherwise all query notes stay on a semitone grid relative to the song and
    an integer-only key search can never be caught out."""
    m = lib[song]
    n = int(rng.integers(12, 21))
    start = int(rng.integers(0, len(m.pitches) - n + 1))
    p = m.pitches[start : start + n].copy()
    d = m.durations[start : start + n].copy()
    true_offset = lead_s + float(m.onsets[start])
    # transposition and tempo
    if fractional:
        p += float(rng.uniform(-7, 7)) + rng.normal(0, intonation_sd, size=n)
    else:
        p += float(rng.integers(-7, 8))
    d /= float(rng.uniform(0.75, 1.3))
    # pitch noise: 25% of notes off by +/- 1 semitone
    k = int(round(0.25 * n))
    idx = rng.choice(n, size=k, replace=False)
    p[idx] += rng.choice([-1.0, 1.0], size=k)
    # one random note split or merge
    i = int(rng.integers(0, n - 1))
    if rng.random() < 0.5:
        p = np.insert(p, i + 1, p[i])
        half = d[i] / 2
        d[i] = half
        d = np.insert(d, i + 1, half)
    else:
        d[i] += d[i + 1]
        p = np.delete(p, i + 1)
        d = np.delete(d, i + 1)
    return Query(song=song, start_note=start, true_offset_s=true_offset, melody=Melody(p, d))


def oracle_pitch(m: Melody, hop_s: float = 0.01, lead_s: float = 0.3, tail_s: float = 0.3) -> np.ndarray:
    """Ideal 10 ms pitch curve for a melody (bypasses audio + CREPE).

    Used to localize failures: if a matcher fails on CREPE pitch but not on
    oracle pitch, the problem is in pitch extraction, not alignment.
    """
    total = lead_s + float(m.durations.sum()) + tail_s
    out = np.full(int(total / hop_s) + 1, np.nan, dtype=np.float32)
    for onset, dur, pitch in zip(m.onsets, m.durations, m.pitches):
        sound = max(dur - ARTICULATION_S, 0.05)
        a = int(round((lead_s + onset) / hop_s))
        out[a : a + int(round(sound / hop_s))] = pitch
    return out
