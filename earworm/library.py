"""Song model and the on-disk index: SQLite metadata + per-stem .npz features."""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import Section, data_path
from .features import Features

SCHEMA = """
CREATE TABLE IF NOT EXISTS songs (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    artist TEXT NOT NULL,
    added_at TEXT,
    audio_path TEXT NOT NULL,
    duration_s REAL NOT NULL,
    source TEXT NOT NULL,
    ingested_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ingest_failures (
    key TEXT PRIMARY KEY,
    title TEXT,
    artist TEXT,
    source TEXT,
    stage TEXT,
    error TEXT,
    at TEXT
);
"""


@dataclass
class Song:
    id: str
    title: str
    artist: str
    added_at: str | None
    audio_path: str
    duration_s: float
    source: str          # "local" | "deezer_preview"


def slugify(text: str) -> str:
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    t = re.sub(r"[^a-zA-Z0-9]+", "-", t.lower()).strip("-")
    return t or "untitled"


def song_id(artist: str, title: str) -> str:
    return f"{slugify(artist)}--{slugify(title)}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Library:
    def __init__(self, cfg: Section):
        self.cfg = cfg
        self.db_path = data_path(cfg, "index.db")
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self):
        self.conn.close()

    # -- paths --
    def features_path(self, sid: str, stem: str) -> Path:
        return data_path(self.cfg, "features", f"{sid}__{stem}.npz")

    def stem_path(self, sid: str, stem: str) -> Path:
        return data_path(self.cfg, "stems", f"{sid}__{stem}.flac")

    @property
    def stems(self) -> list[str]:
        return list(self.cfg.separation.stems)

    def has_features(self, sid: str) -> bool:
        return all(self.features_path(sid, st).exists() for st in self.stems) and self.get(sid) is not None

    # -- songs --
    def upsert(self, song: Song) -> None:
        d = asdict(song)
        d["ingested_at"] = _now()
        self.conn.execute(
            "INSERT INTO songs (id,title,artist,added_at,audio_path,duration_s,source,ingested_at) "
            "VALUES (:id,:title,:artist,:added_at,:audio_path,:duration_s,:source,:ingested_at) "
            "ON CONFLICT(id) DO UPDATE SET title=excluded.title, artist=excluded.artist, "
            "added_at=excluded.added_at, audio_path=excluded.audio_path, duration_s=excluded.duration_s, "
            "source=excluded.source, ingested_at=excluded.ingested_at", d)
        self.conn.execute("DELETE FROM ingest_failures WHERE key=?", (song.id,))
        self.conn.commit()

    def get(self, sid: str) -> Song | None:
        r = self.conn.execute("SELECT * FROM songs WHERE id=?", (sid,)).fetchone()
        return self._row(r) if r else None

    def songs(self) -> list[Song]:
        return [self._row(r) for r in self.conn.execute("SELECT * FROM songs ORDER BY artist, title")]

    @staticmethod
    def _row(r) -> Song:
        return Song(id=r["id"], title=r["title"], artist=r["artist"], added_at=r["added_at"],
                    audio_path=r["audio_path"], duration_s=r["duration_s"], source=r["source"])

    # -- failures --
    def record_failure(self, key: str, title: str, artist: str, source: str, stage: str, error: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO ingest_failures (key,title,artist,source,stage,error,at) VALUES (?,?,?,?,?,?,?)",
            (key, title, artist, source, stage, error[:2000], _now()))
        self.conn.commit()

    def failures(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM ingest_failures ORDER BY at")]

    # -- features --
    def save_features(self, sid: str, stem: str, f: Features) -> None:
        f.save(self.features_path(sid, stem))

    def load_features(self, sid: str) -> dict[str, Features]:
        out = {}
        for st in self.stems:
            p = self.features_path(sid, st)
            if p.exists():
                out[st] = Features.load(p)
        return out

    def entries(self):
        """All songs with features, as search entries."""
        from .search import Entry

        out = []
        for s in self.songs():
            stems = self.load_features(s.id)
            if stems:
                out.append(Entry(song_id=s.id, title=s.title, artist=s.artist, stems=stems,
                                 added_at=s.added_at, meta={"source": s.source, "audio_path": s.audio_path,
                                                             "duration_s": s.duration_s}))
        return out
