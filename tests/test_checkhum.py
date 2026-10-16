import numpy as np
import soundfile as sf

import synth
from earworm import cli
from earworm.checkhum import format_report, hum_report


def _melody(seed=3):
    return synth.random_melody(np.random.default_rng(seed), 25, 30)


def _write(tmp_path, name, audio):
    p = tmp_path / name
    sf.write(p, audio.astype(np.float32), 16000, subtype="FLOAT")   # float wav: values above 1.0 survive
    return str(p)


def test_good_hum_is_ok(tmp_path, cfg):
    rep = hum_report(_write(tmp_path, "good.wav", synth.render(_melody())), cfg)
    assert rep["warnings"] == [], rep
    assert rep["n_notes"] >= 20 and rep["voiced_fraction"] > 0.5 and rep["clipped_fraction"] == 0.0
    assert "OK" in format_report(rep)


def test_silence_and_noise_warn(tmp_path, cfg):
    sil = hum_report(_write(tmp_path, "silence.wav", np.zeros(16000 * 8)), cfg)
    assert sil["n_notes"] == 0 and any("quiet" in w for w in sil["warnings"])
    assert any("few notes" in w for w in sil["warnings"]) and any("little voiced" in w for w in sil["warnings"])
    noise = np.random.default_rng(0).normal(0, 0.05, 16000 * 8)
    rep = hum_report(_write(tmp_path, "noise.wav", noise), cfg)
    assert any("unvoiced" in w or "few notes" in w or "flat pitch" in w for w in rep["warnings"]), rep


def test_clipped_hum_warns(tmp_path, cfg):
    loud = np.clip(synth.render(_melody()) * 8.0, -1.0, 1.0)
    rep = hum_report(_write(tmp_path, "clipped.wav", loud), cfg)
    assert any("clipped" in w for w in rep["warnings"]) and rep["clipped_fraction"] > 0.01


def test_short_hum_warns(tmp_path, cfg):
    audio = synth.render(_melody())[: 16000 * 1]
    rep = hum_report(_write(tmp_path, "short.wav", audio), cfg)
    assert any(w.startswith("short") for w in rep["warnings"]), rep


def test_quiet_hum_warns(tmp_path, cfg):
    rep = hum_report(_write(tmp_path, "quiet.wav", synth.render(_melody()) * 0.001), cfg)
    assert any("quiet" in w for w in rep["warnings"]), rep


def test_cli_exit_codes_and_summary(tmp_path, capsys):
    good = _write(tmp_path, "good.wav", synth.render(_melody()))
    bad = tmp_path / "broken.wav"
    bad.write_bytes(b"not audio")
    assert cli.main(["check-hum", good]) == 0
    assert "OK" in capsys.readouterr().out
    assert cli.main(["check-hum", good, str(bad)]) == 1            # unreadable file -> non-zero
    out = capsys.readouterr().out
    assert "ERROR" in out and "1 OK, 0 with warnings, 1 unreadable" in out
    assert cli.main(["check-hum", str(tmp_path / "missing.wav")]) == 1
