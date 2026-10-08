"""Per-song ingestion: separate -> pitch -> notes -> features on disk + index row."""

from __future__ import annotations

import logging
import traceback
from typing import Iterable

from .config import Section
from .features import features_from_audio
from .library import Library, Song

log = logging.getLogger(__name__)


def ingest_song(song: Song, library: Library, cfg: Section, force: bool = False) -> bool:
    """Returns True if the song was (re)processed, False if skipped as already done."""
    from .separate import separate_cached

    if not force and library.has_features(song.id):
        return False
    stage = "separate"
    try:
        stems = separate_cached(song.id, song.audio_path, cfg, library, force=force)
        stage = "features"
        for stem, audio in stems.items():
            library.save_features(song.id, stem, features_from_audio(audio, cfg))
        if not song.duration_s:
            any_stem = next(iter(stems.values()))
            song.duration_s = len(any_stem) / float(cfg.audio.sample_rate)
        library.upsert(song)
        return True
    except Exception as e:
        log.error("ingest failed at %s for %s: %s", stage, song.id, e)
        log.debug(traceback.format_exc())
        library.record_failure(song.id, song.title, song.artist, song.source, stage, f"{type(e).__name__}: {e}")
        return False


def ingest_songs(songs: Iterable[Song], library: Library, cfg: Section, force: bool = False) -> dict:
    counts = {"processed": 0, "skipped": 0, "failed": 0}
    for s in songs:
        if not force and library.has_features(s.id):
            counts["skipped"] += 1
            continue
        log.info("ingesting %s - %s", s.artist, s.title)
        if ingest_song(s, library, cfg, force=force):
            counts["processed"] += 1
        else:
            counts["failed"] += 1
    return counts
