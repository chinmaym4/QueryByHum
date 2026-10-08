import csv

import pytest

from earworm.library import song_id
from earworm.sources.deezer import DeezerClient, fetch_previews, match_score
from earworm.sources.local import parse_filename
from earworm.sources.playlist_csv import PlaylistRow, read_playlist


def test_parse_filename(tmp_path):
    assert parse_filename(tmp_path / "Drake - Hotline Bling.mp3") == ("Drake", "Hotline Bling")
    assert parse_filename(tmp_path / "untitled track.mp3") == (None, "untitled track")


def test_song_id_is_stable_slug():
    assert song_id("Beyoncé", "Halo (Live)") == "beyonce--halo-live"


def test_read_plain_and_exportify_csv(tmp_path):
    p = tmp_path / "plain.csv"
    p.write_text("title,artist,added_at\nHotline Bling,Drake,2020-01-01\n,Nobody,\n")
    rows = read_playlist(p)
    assert rows == [PlaylistRow("Hotline Bling", "Drake", "2020-01-01")]

    e = tmp_path / "exportify.csv"
    with open(e, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Track URI", "Track Name", "Album Name", "Artist Name(s)", "Release Date", "Added At"])
        w.writerow(["spotify:track:x", "Sicko Mode", "Astroworld", "Travis Scott;Drake", "2018", "2019-05-02T10:00:00Z"])
    rows = read_playlist(e)
    assert rows[0].title == "Sicko Mode" and rows[0].primary_artist == "Travis Scott"
    assert rows[0].added_at == "2019-05-02T10:00:00Z"


def test_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("song,singer\nx,y\n")
    with pytest.raises(ValueError, match="missing column"):
        read_playlist(p)


def test_match_score():
    row = PlaylistRow("Hotline Bling", "Drake", None)
    assert match_score(row, "Hotline Bling", "Drake") == 100
    assert match_score(row, "Hotline Bling (Remix)", "Drake") >= 85
    assert match_score(row, "Bling Bling", "Lil Wayne") < 85


class FakeResp:
    def __init__(self, status=200, json=None, content=b""):
        self.status_code, self._json, self.content = status, json, content
        self.text = str(json)

    def json(self):
        if self._json is None:
            raise ValueError("not json")
        return self._json


class FakeSession:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return self.routes(url, params)


def test_fetch_previews_matches_downloads_and_logs_unmatched(cfg, tmp_path):
    import copy
    cfg = copy.deepcopy(cfg)
    cfg.paths.data_dir = str(tmp_path)
    cfg.deezer.requests_per_second = 1000

    def routes(url, params):
        if url == cfg.deezer.search_url:
            q = params["q"]
            if "Hotline" in q:
                return FakeResp(json={"data": [
                    {"id": 1, "title": "Bling Bling", "artist": {"name": "Other"}, "preview": "http://p/0.mp3"},
                    {"id": 2, "title": "Hotline Bling", "title_short": "Hotline Bling",
                     "artist": {"name": "Drake"}, "preview": "http://p/2.mp3"}]})
            if "Broken" in q:
                return FakeResp(json={"error": {"type": "Exception", "message": "Quota limit exceeded", "code": 4}})
            return FakeResp(json={"data": []})
        if url == "http://p/2.mp3":
            return FakeResp(content=b"ID3fake")
        return FakeResp(status=404)

    sess = FakeSession(routes)
    rows = [PlaylistRow("Hotline Bling", "Drake", "2020-01-01"), PlaylistRow("Nope", "Nobody", None),
            PlaylistRow("Broken", "Api", None)]
    songs = fetch_previews(rows, cfg, client=DeezerClient(cfg, session=sess))
    assert [s.id for s in songs] == ["drake--hotline-bling"]
    assert songs[0].source == "deezer_preview"
    assert (tmp_path / "audio" / "drake--hotline-bling.mp3").read_bytes() == b"ID3fake"
    assert sess.calls[0][1]["q"] == 'artist:"Drake" track:"Hotline Bling"'
    un = list(csv.DictReader(open(tmp_path / "unmatched.csv")))
    assert {u["title"]: u["reason"] for u in un} == {
        "Nope": "no results", "Broken": un[1]["reason"]}
    assert "Quota" in un[1]["reason"]
