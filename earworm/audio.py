"""ffmpeg decoding. All audio entering earworm goes through `decode`.

Pitch extraction always uses 16 kHz mono float32; demucs uses 44.1 kHz stereo.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np


class AudioDecodeError(RuntimeError):
    pass


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise AudioDecodeError("ffmpeg not found on PATH (check `ffmpeg -version`)")
    return exe


def decode(path: str | Path, sample_rate: int = 16000, channels: int = 1) -> np.ndarray:
    """Decode any ffmpeg-readable file to float32.

    Returns shape (n,) for mono, (channels, n) otherwise.
    """
    path = Path(path)
    if not path.is_file():
        raise AudioDecodeError(f"no such file: {path}")
    cmd = [
        _ffmpeg(), "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(path), "-vn",
        "-ac", str(channels), "-ar", str(sample_rate),
        "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise AudioDecodeError(f"ffmpeg failed on {path}: {proc.stderr.decode(errors='replace').strip()}")
    audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        raise AudioDecodeError(f"ffmpeg decoded no samples from {path}")
    if channels > 1:
        audio = audio.reshape(-1, channels).T.copy()
    return audio


def decode_mono16k(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    return normalize(decode(path, sample_rate=sample_rate, channels=1))


def normalize(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """Peak-normalize; silent input is returned unchanged."""
    audio = np.asarray(audio, dtype=np.float32)
    m = float(np.max(np.abs(audio))) if audio.size else 0.0
    if m < 1e-8:
        return audio
    return (audio * (peak / m)).astype(np.float32)


def trim_silence(audio: np.ndarray, top_db: float) -> np.ndarray:
    import librosa

    if audio.size == 0:
        return audio
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed.astype(np.float32)


def write_flac(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sample_rate, format="FLAC")


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return audio, sr
