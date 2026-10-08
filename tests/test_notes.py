import numpy as np

import synth
from earworm.notes import segment_notes
from earworm.pitch import clean_pitch, run_crepe


def test_segmentation_on_ideal_curve(cfg):
    m = synth.Melody(np.array([60, 60, 62, 64, 64, 59.0]), np.array([0.3, 0.5, 0.3, 0.6, 0.4, 0.8]))
    notes = segment_notes(synth.oracle_pitch(m), cfg)
    assert len(notes) == 6
    np.testing.assert_allclose(notes[:, 2], m.pitches)
    np.testing.assert_allclose(notes[:, 0], m.onsets + 0.3, atol=0.011)


def test_legato_pitch_change_splits_note(cfg):
    p = np.full(100, 60.0)
    p[50:] = 62.0
    p[20:23] = 63.0     # 30 ms blip: not a new note
    notes = segment_notes(p, cfg)
    assert len(notes) == 2
    np.testing.assert_allclose(notes[:, 2], [60, 62])
    np.testing.assert_allclose(notes[:, 0], [0.0, 0.5], atol=1e-6)


def test_short_note_merges_into_closest_neighbor(cfg):
    p = np.full(100, 60.0)
    p[40:46] = 64.5     # 60 ms note (> 40 ms so it splits, < 80 ms so it merges)
    p[46:] = 65.0
    notes = segment_notes(p, cfg)
    assert len(notes) == 2
    np.testing.assert_allclose(notes[1, 0], 0.40, atol=1e-6)   # merged into the 65 note
    np.testing.assert_allclose(notes[:, 2], [60, 65])


def test_segmentation_on_rendered_sine_sequence(cfg):
    m = synth.Melody(np.array([57, 60, 60, 64, 62, 67, 65.0]), np.array([0.4, 0.3, 0.5, 0.4, 0.3, 0.6, 0.5]))
    hz, per = run_crepe(synth.render(m), cfg)
    notes = segment_notes(clean_pitch(hz, per, cfg), cfg)
    assert len(notes) == len(m.pitches)
    np.testing.assert_allclose(notes[:, 2], m.pitches, atol=0.3)
    np.testing.assert_allclose(notes[:, 0], m.onsets + 0.3, atol=0.05)
