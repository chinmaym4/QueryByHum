"""demucs wrapper with stem caching (16 kHz mono FLAC in data/stems/)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import Section, resolve_device

_MODELS: dict = {}


def _model(name: str, device: str):
    key = (name, device)
    if key not in _MODELS:
        from demucs.pretrained import get_model

        m = get_model(name)
        m.to(device).eval()
        _MODELS[key] = m
    return _MODELS[key]


def separate(audio_path: str | Path, cfg: Section) -> dict[str, np.ndarray]:
    """Decode to 44.1 kHz stereo, run demucs, return requested stems as 16 kHz mono."""
    import librosa
    import torch
    from demucs.apply import apply_model

    from .audio import decode

    device = resolve_device(cfg)
    model = _model(cfg.separation.model, device)
    sr_in = int(cfg.audio.separation_sample_rate)
    if model.samplerate != sr_in:
        raise ValueError(f"demucs model expects {model.samplerate} Hz, config says {sr_in}")
    wav = decode(audio_path, sample_rate=sr_in, channels=model.audio_channels)
    x = torch.from_numpy(wav)
    ref = x.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    x = (x - mean) / std
    with torch.inference_mode():
        out = apply_model(model, x[None], device=device, split=bool(cfg.separation.segment_split),
                          overlap=float(cfg.separation.overlap), progress=False)[0]
    out = out * std + mean
    stems = {}
    for st in cfg.separation.stems:
        if st not in model.sources:
            raise ValueError(f"stem {st!r} not produced by {cfg.separation.model} (has {model.sources})")
        mono = out[model.sources.index(st)].mean(0).cpu().numpy()
        stems[st] = librosa.resample(mono, orig_sr=sr_in, target_sr=int(cfg.audio.sample_rate)).astype(np.float32)
    return stems


def separate_cached(sid: str, audio_path: str | Path, cfg: Section, library, force: bool = False) -> dict[str, np.ndarray]:
    from .audio import read_audio, write_flac

    paths = {st: library.stem_path(sid, st) for st in cfg.separation.stems}
    if not force and all(p.exists() for p in paths.values()):
        return {st: read_audio(p)[0] for st, p in paths.items()}
    stems = separate(audio_path, cfg)
    for st, a in stems.items():
        write_flac(paths[st], a, int(cfg.audio.sample_rate))
    return stems
