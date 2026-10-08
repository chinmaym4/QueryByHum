"""Ingest a folder of audio files. Title/artist from tags, else `Artist - Title` filename."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from ..library import Song, song_id

log = logging.getLogger(__name__)

AUDIO_EXTS = {".mp3", ".m4a", ".mp4", ".aac", ".flac", ".wav", ".ogg", ".opus", ".webm", ".aiff", ".aif", ".wma"}


def _first(tags, key) -> str | None:
    if not tags:
        return None
    v = tags.get(key)
    if isinstance(v, list):
        v = v[0] if v else None
    v = str(v).strip() if v is not None else None
    return v or None


def read_tags(path: Path) -> tuple[str | None, str | None, float | None]:
    import mutagen

    try:
        f = mutagen.File(str(path), easy=True)
    except Exception as e:  # corrupt or unsupported container
        log.warning("could not read tags from %s: %s", path, e)
        return None, None, None
    if f is None:
        return None, None, None
    duration = float(f.info.length) if getattr(f, "info", None) and getattr(f.info, "length", None) else None
    return _first(f.tags, "title"), _first(f.tags, "artist"), duration


def parse_filename(path: Path) -> tuple[str | None, str]:
    stem = path.stem.strip()
    if " - " in stem:
        artist, title = stem.split(" - ", 1)
        return artist.strip() or None, title.strip()
    return None, stem


def scan_folder(folder: str | Path) -> Iterator[Song]:
    folder = Path(folder)
    for p in sorted(folder.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in AUDIO_EXTS:
            continue
        title, artist, duration = read_tags(p)
        f_artist, f_title = parse_filename(p)
        title = title or f_title
        artist = artist or f_artist or "Unknown Artist"
        yield Song(id=song_id(artist, title), title=title, artist=artist, added_at=None,
                   audio_path=str(p.resolve()), duration_s=duration or 0.0, source="local")
