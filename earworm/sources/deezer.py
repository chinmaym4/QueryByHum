"""Fetch 30 s Deezer preview MP3s for playlist rows.

Uses the public search API (no auth). The endpoint/shape is verified at runtime:
anything unexpected is logged and the song is skipped, never guessed past.
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from ..config import Section, data_path
from ..library import Song, song_id
from .playlist_csv import PlaylistRow

log = logging.getLogger(__name__)


class DeezerError(RuntimeError):
    pass


class RateLimiter:
    def __init__(self, per_second: float):
        self.min_interval = 1.0 / per_second if per_second > 0 else 0.0
        self._last = 0.0

    def wait(self):
        dt = time.monotonic() - self._last
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)
        self._last = time.monotonic()


@dataclass
class DeezerHit:
    deezer_id: int
    title: str
    artist: str
    preview_url: str
    score: float


def match_score(row: PlaylistRow, title: str, artist: str) -> float:
    """Mean of title and artist similarity (rapidfuzz token_set_ratio, 0..100)."""
    from rapidfuzz import fuzz

    t = fuzz.token_set_ratio(row.title.lower(), title.lower())
    a = max(fuzz.token_set_ratio(row.artist.lower(), artist.lower()),
            fuzz.token_set_ratio(row.primary_artist.lower(), artist.lower()))
    return (t + a) / 2.0


class DeezerClient:
    def __init__(self, cfg: Section, session: requests.Session | None = None):
        self.cfg = cfg
        self.c = cfg.deezer
        self.http = session or requests.Session()
        self.limiter = RateLimiter(float(self.c.requests_per_second))

    def _get(self, url: str, **kw) -> requests.Response:
        self.limiter.wait()
        return self.http.get(url, timeout=float(self.c.timeout_s), **kw)

    def search(self, row: PlaylistRow) -> list[dict]:
        queries = [f'artist:"{row.primary_artist}" track:"{row.title}"', f"{row.primary_artist} {row.title}"]
        for q in queries:
            r = self._get(self.c.search_url, params={"q": q})
            if r.status_code != 200:
                raise DeezerError(f"search HTTP {r.status_code}: {r.text[:200]}")
            try:
                body = r.json()
            except ValueError as e:
                raise DeezerError(f"search returned non-JSON: {r.text[:200]}") from e
            if "error" in body:
                raise DeezerError(f"search API error: {body['error']}")
            if "data" not in body:
                raise DeezerError(f"unexpected search response shape (keys: {list(body)})")
            if body["data"]:
                return body["data"]
        return []

    def best_match(self, row: PlaylistRow) -> DeezerHit | None:
        best = None
        for item in self.search(row):
            artist = (item.get("artist") or {}).get("name", "")
            titles = [item.get("title") or "", item.get("title_short") or ""]
            score = max(match_score(row, t, artist) for t in titles if t) if any(titles) else 0.0
            if best is None or score > best.score:
                best = DeezerHit(int(item.get("id", 0)), item.get("title", ""), artist,
                                 item.get("preview") or "", score)
        return best

    def download(self, url: str, dest: Path) -> None:
        r = self._get(url)
        if r.status_code != 200 or not r.content:
            raise DeezerError(f"preview download HTTP {r.status_code} ({len(r.content)} bytes)")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)


UNMATCHED_FIELDS = ["title", "artist", "added_at", "reason", "best_title", "best_artist", "best_score"]


def fetch_previews(rows: list[PlaylistRow], cfg: Section, library=None, force: bool = False,
                   client: DeezerClient | None = None) -> list[Song]:
    """Resolve rows to downloaded previews. Unmatched rows go to data/unmatched.csv."""
    client = client or DeezerClient(cfg)
    threshold = float(cfg.deezer.match_threshold)
    songs: list[Song] = []
    unmatched: list[dict] = []
    for row in rows:
        sid = song_id(row.primary_artist, row.title)
        dest = data_path(cfg, "audio", f"{sid}.mp3")
        rec = {"title": row.title, "artist": row.artist, "added_at": row.added_at or ""}
        if library is not None and not force and library.has_features(sid):
            songs.append(library.get(sid))
            continue
        try:
            hit = client.best_match(row)
        except (DeezerError, requests.RequestException) as e:
            log.error("Deezer search failed for %s - %s: %s", row.artist, row.title, e)
            unmatched.append({**rec, "reason": f"search failed: {e}"})
            continue
        if hit is None:
            unmatched.append({**rec, "reason": "no results"})
            continue
        if hit.score < threshold or not hit.preview_url:
            reason = "no preview url" if hit.score >= threshold else f"best score {hit.score:.0f} < {threshold:.0f}"
            unmatched.append({**rec, "reason": reason, "best_title": hit.title,
                              "best_artist": hit.artist, "best_score": f"{hit.score:.1f}"})
            continue
        if force or not dest.exists():
            try:
                client.download(hit.preview_url, dest)
            except (DeezerError, requests.RequestException) as e:
                log.error("preview download failed for %s - %s: %s", row.artist, row.title, e)
                unmatched.append({**rec, "reason": f"download failed: {e}"})
                continue
        songs.append(Song(id=sid, title=row.title, artist=row.primary_artist, added_at=row.added_at,
                          audio_path=str(dest), duration_s=30.0, source="deezer_preview"))
    path = data_path(cfg, "unmatched.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=UNMATCHED_FIELDS)
        w.writeheader()
        for u in unmatched:
            w.writerow({k: u.get(k, "") for k in UNMATCHED_FIELDS})
    if unmatched:
        log.warning("%d of %d rows unmatched; see %s", len(unmatched), len(rows), path)
    return songs
