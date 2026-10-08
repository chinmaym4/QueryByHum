import numpy as np

from earworm.pitch import (clean_pitch, downsample_nanmedian, drop_short_runs, fix_octave_errors,
                           hz_to_midi, median_filter_runs, midi_to_hz)


def test_hz_to_midi_known_values():
    np.testing.assert_allclose(hz_to_midi(np.array([440.0, 220.0, 880.0, 261.6256])), [69, 57, 81, 60], atol=1e-4)
    assert np.isnan(hz_to_midi(np.array([0.0, np.nan]))).all()


def test_hz_midi_roundtrip():
    m = np.linspace(36, 84, 97)
    np.testing.assert_allclose(hz_to_midi(midi_to_hz(m)), m, atol=1e-9)


def test_octave_fix_folds_short_excursion():
    x = np.full(60, 60.0)
    x[20:30] += 12.0          # 100 ms up an octave, then back
    x[40:44] -= 12.0          # 40 ms down an octave, then back
    out = fix_octave_errors(x, 11, 13, max_frames=15)
    np.testing.assert_allclose(out, 60.0)


def test_octave_fix_keeps_long_excursion_and_real_leaps():
    x = np.full(60, 60.0)
    x[10:40] += 12.0          # 300 ms: a real octave leap, not an error
    np.testing.assert_allclose(fix_octave_errors(x, 11, 13, 15), x)
    y = np.full(40, 60.0)
    y[20:] = 72.0             # leap without return
    np.testing.assert_allclose(fix_octave_errors(y, 11, 13, 15), y)
    z = np.full(40, 60.0)
    z[20:25] = 67.0           # a fifth: not an octave error
    np.testing.assert_allclose(fix_octave_errors(z, 11, 13, 15), z)


def test_median_filter_does_not_cross_gaps():
    x = np.array([60, 60, 61, 60, 60, np.nan, 70, 70, 70, 70, 70.0])
    out = median_filter_runs(x, 5)
    np.testing.assert_allclose(out[:5], 60)
    assert np.isnan(out[5])
    np.testing.assert_allclose(out[6:], 70)


def test_drop_short_runs():
    x = np.full(20, np.nan)
    x[2:6] = 60    # 40 ms -> dropped
    x[10:15] = 62  # 50 ms -> kept
    out = drop_short_runs(x, 5)
    assert np.isnan(out[2:6]).all()
    np.testing.assert_allclose(out[10:15], 62)


def test_downsample_nanmedian():
    x = np.array([1, 2, 3, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 5, 5], dtype=float)
    out = downsample_nanmedian(x, 5)
    np.testing.assert_allclose(out[0], 2)
    assert np.isnan(out[1])
    np.testing.assert_allclose(out[2], 5)


def test_clean_pitch_voicing_threshold(cfg):
    hz = np.full(30, 440.0)
    per = np.full(30, 0.9)
    per[10:20] = 0.1
    out = clean_pitch(hz, per, cfg)
    assert np.isnan(out[10:20]).all()
    np.testing.assert_allclose(out[:10], 69, atol=1e-4)
