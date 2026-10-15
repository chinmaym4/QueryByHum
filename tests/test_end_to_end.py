"""CLI end-to-end on rendered audio files: ingest -> search -> explain -> eval -> stats.

demucs is stubbed (its weights are a download, and separation quality is not
what these tests check); everything else is the real pipeline incl. ffmpeg + CREPE.
"""

import csv
import re

import numpy as np
import pytest
import soundfile as sf

import synth
from earworm import cli
from earworm.config import find_config_path


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    cfg_text = find_config_path().read_text()
    cfg_text = re.sub(r'data_dir = ".*"', f'data_dir = "{tmp_path / "data"}"', cfg_text)
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(cfg_text)

    from earworm import separate

    def fake_separate(path, cfg):
        from earworm.audio import decode

        a = decode(path, sample_rate=16000)
        return {"vocals": a, "other": np.zeros_like(a)}

    monkeypatch.setattr(separate, "separate", fake_separate)

    rng = np.random.default_rng(5)
    music = tmp_path / "music"
    music.mkdir()
    melodies = {}
    for k in range(4):
        m = synth.random_melody(rng, 30, 34)
        name = f"Artist {k} - Song {k}.wav"
        sf.write(music / name, synth.render(m), 16000)
        melodies[k] = m
    return tmp_path, cfg_file, melodies, rng


def run(cfg_file, *args):
    return cli.main(["--config", str(cfg_file), *args])


def test_cli_end_to_end(workspace, capsys):
    tmp, cfg_file, melodies, rng = workspace
    assert run(cfg_file, "ingest", "--folder", str(tmp / "music")) == 0
    out = capsys.readouterr().out
    assert "processed=4" in out

    # idempotent
    assert run(cfg_file, "ingest", "--folder", str(tmp / "music")) == 0
    assert "skipped=4" in capsys.readouterr().out

    # hum an excerpt of song 2, transposed and slower, as an mp3 (exercises ffmpeg decode)
    m = melodies[2]
    q = synth.Melody(m.pitches[8:22] - 3, m.durations[8:22] * 1.15)
    wav = tmp / "hum.wav"
    sf.write(wav, synth.render(q), 16000)
    hum = tmp / "hum.mp3"
    import subprocess
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(wav), str(hum)], check=True)

    assert run(cfg_file, "search", str(hum), "--k", "3") == 0
    lines = capsys.readouterr().out.splitlines()
    assert "Song 2" in lines[1], "\n".join(lines)

    assert run(cfg_file, "search", str(hum), "--matcher", "contour", "--artist", "Artist 1") == 0
    capsys.readouterr()

    png = tmp / "explain.png"
    assert run(cfg_file, "explain", str(hum), "artist-2--song-2", "--out", str(png)) == 0
    assert png.exists() and png.stat().st_size > 10_000

    ev = tmp / "eval"
    (ev / "hums").mkdir(parents=True)
    hum.rename(ev / "hums" / "hum01.mp3")
    with open(ev / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hum_file", "title", "artist", "part", "in_preview", "notes", "true_offset_s"])
        w.writerow(["hum01.mp3", "Song 2", "Artist 2", "chorus", "y", "", f"{0.3 + m.onsets[8]:.3f}"])
        w.writerow(["hum02.mp3", "Not In Library", "Nobody", "verse", "unknown", "", ""])
    assert run(cfg_file, "eval", "--eval-dir", str(ev), "--weights-search") == 0
    out = capsys.readouterr().out
    assert "EXCLUDED: hum02.mp3" in out and "song not in library" in out
    assert "Leave-one-out" in out and "without dtw" in out
    assert "excluded 1 not in library, 0 unreadable" in out
    assert "Offset accuracy (seconds into the indexed audio" in out and "max spread between matchers" in out
    block = out.splitlines()
    start = next(i for i, l in enumerate(block) if "all hums with true_offset_s" in l)
    fused_line = next(l for l in block[start:] if l.strip().startswith("fused"))
    assert float(fused_line.split()[1]) == 1.0      # fused offset within 1 s of the true start
    assert "WARNING" in out
    results = list((ev / "results").glob("*.csv"))
    assert len(results) == 3                            # per-hum, metrics, offset accuracy
    per_hum = next(p for p in results if not p.stem.endswith(("_metrics", "_offset_accuracy")))
    rows = list(csv.DictReader(open(per_hum)))
    assert rows[0]["rank_fused"] == "1"

    assert run(cfg_file, "stats", "--list") == 0
    out = capsys.readouterr().out
    assert "library: 4 songs" in out and "local: 4" in out
