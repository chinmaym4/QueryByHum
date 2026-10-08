"""Per-recording feature bundle shared by songs (per stem) and queries."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Section


@dataclass(eq=False)
class Features:
    pitch_10ms: np.ndarray            # MIDI, NaN where unvoiced, 10 ms hop
    pitch_50ms: np.ndarray            # nanmedian over 5 frames
    notes: np.ndarray                 # float32[n, 3]: onset_s, duration_s, midi_pitch
    hop_s: float = 0.01
    downsample_factor: int = 5
    cache: dict = field(default_factory=dict, repr=False)  # matcher-derived data

    @property
    def hop_50ms(self) -> float:
        return self.hop_s * self.downsample_factor

    @property
    def voiced_duration_s(self) -> float:
        return float(np.isfinite(self.pitch_50ms).sum()) * self.hop_50ms

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, pitch_10ms=self.pitch_10ms.astype(np.float32),
            pitch_50ms=self.pitch_50ms.astype(np.float32),
            notes=self.notes.astype(np.float32),
            hop_s=np.float32(self.hop_s), downsample_factor=np.int32(self.downsample_factor),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Features":
        with np.load(path) as z:
            return cls(
                pitch_10ms=z["pitch_10ms"], pitch_50ms=z["pitch_50ms"], notes=z["notes"],
                hop_s=float(z["hop_s"]), downsample_factor=int(z["downsample_factor"]),
            )


def features_from_pitch(pitch_10ms: np.ndarray, cfg: Section) -> Features:
    from .notes import segment_notes
    from .pitch import downsample_nanmedian

    factor = int(cfg.pitch.downsample_factor)
    return Features(
        pitch_10ms=np.asarray(pitch_10ms, dtype=np.float32),
        pitch_50ms=downsample_nanmedian(pitch_10ms, factor),
        notes=segment_notes(pitch_10ms, cfg),
        hop_s=float(cfg.pitch.hop_s), downsample_factor=factor,
    )


def features_from_audio(audio: np.ndarray, cfg: Section, is_query: bool = False) -> Features:
    """16 kHz mono audio -> Features. Queries get leading/trailing silence trimmed."""
    from .audio import normalize, trim_silence
    from .pitch import extract_pitch, trim_unvoiced_edges

    audio = normalize(audio)
    if is_query:
        audio = trim_silence(audio, cfg.audio.query_trim_top_db)
    p10, _ = extract_pitch(audio, cfg)
    if is_query:
        p10, _ = trim_unvoiced_edges(p10)
    return features_from_pitch(p10, cfg)
