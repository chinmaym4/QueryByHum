"""Read a playlist CSV: title,artist[,added_at], or an Exportify export."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

COLUMN_ALIASES = {
    "title": ("title", "track name", "track", "name"),
    "artist": ("artist", "artist name(s)", "artists", "artist name"),
    "added_at": ("added_at", "added at", "date added"),
}


@dataclass
class PlaylistRow:
    title: str
    artist: str            # as given (may list several artists)
    added_at: str | None

    @property
    def primary_artist(self) -> str:
        """First listed artist. Exportify separates artists with ';' (older exports used ',')."""
        for sep in (";", ","):
            if sep in self.artist:
                return self.artist.split(sep)[0].strip()
        return self.artist.strip()


def _resolve_columns(header: list[str]) -> dict[str, str]:
    lower = {h.strip().lower(): h for h in header}
    out = {}
    for key, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            if a in lower:
                out[key] = lower[a]
                break
    missing = {"title", "artist"} - out.keys()
    if missing:
        raise ValueError(f"CSV is missing column(s) {sorted(missing)}; header was {header}")
    return out


def read_playlist(path: str | Path) -> list[PlaylistRow]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = _resolve_columns(reader.fieldnames or [])
        rows = []
        for r in reader:
            title = (r.get(cols["title"]) or "").strip()
            artist = (r.get(cols["artist"]) or "").strip()
            if not title or not artist:
                continue
            added = (r.get(cols["added_at"]) or "").strip() if "added_at" in cols else ""
            rows.append(PlaylistRow(title=title, artist=artist, added_at=added or None))
    return rows
