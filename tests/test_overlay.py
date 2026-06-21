import unittest

import numpy as np

from mai.overlay import (
    RegionDescriptor,
    best_overlay,
    foote_novelty_boundaries,
    optimal_beat_offset,
    overlay_score,
    plan_mix,
    spectral_complementarity,
)


def _region(track_id='A', position=0.5, tempo=120.0, key=0, energy=0.6,
            vocal=0.0, bands=(0.34, 0.33, 0.33), onset=None, bars=8, drop=False):
    if onset is None:
        onset = np.zeros(40); onset[10] = 1.0; onset[18] = 1.0; onset[26] = 1.0
    return RegionDescriptor(
        track_id=track_id, start_s=0.0, end_s=16.0, position=position, tempo=tempo,
        key=key, energy=energy, vocal_activity=vocal,
        band_profile=np.asarray(bands, dtype=float), onset_envelope=np.asarray(onset, dtype=float),
        bars=bars, is_drop=drop,
    )


class BeatOffsetTests(unittest.TestCase):
    def test_identical_onsets_align_at_zero(self):
        onset = np.zeros(50); onset[[8, 16, 24, 32]] = 1.0
        lag, strength = optimal_beat_offset(onset, onset)
        self.assertEqual(lag, 0)
        self.assertGreater(strength, 0.95)

    def test_shifted_onsets_recover_lag(self):
        a = np.zeros(50); a[[10, 20, 30]] = 1.0
        b = np.zeros(50); b[[13, 23, 33]] = 1.0  # same pattern, +3 frames
        lag, strength = optimal_beat_offset(a, b)
        self.assertEqual(abs(lag), 3)
        self.assertGreater(strength, 0.5)

    def test_empty_is_safe(self):
        self.assertEqual(optimal_beat_offset(np.array([]), np.array([1.0])), (0, 0.0))


class ComplementarityTests(unittest.TestCase):
    def test_disjoint_bands_complement(self):
        self.assertGreater(spectral_complementarity([0.9, 0.05, 0.05], [0.05, 0.05, 0.9]), 0.7)

    def test_identical_bands_collide(self):
        self.assertLess(spectral_complementarity([0.34, 0.33, 0.33], [0.34, 0.33, 0.33]), 0.05)


class OverlayScoreTests(unittest.TestCase):
    def test_complementary_instrumental_scores_high(self):
        good = overlay_score(_region(bands=(0.7, 0.2, 0.1)), _region(bands=(0.1, 0.2, 0.7)))
        bad = overlay_score(_region(bands=(0.7, 0.2, 0.1), tempo=120),
                            _region(bands=(0.7, 0.2, 0.1), tempo=150, vocal=1.0))
        self.assertGreater(good.score, bad.score)

    def test_two_vocals_force_cut(self):
        match = overlay_score(_region(vocal=0.9), _region(vocal=0.9))
        self.assertEqual(match.blend_type, 'cut')
        self.assertLess(match.components['vocal_compat'], 0.5)

    def test_tempo_clash_echoes_out(self):
        match = overlay_score(_region(tempo=120.0), _region(tempo=92.0))
        self.assertEqual(match.blend_type, 'echo_out')

    def test_double_drop_detected(self):
        match = overlay_score(
            _region(tempo=128, bands=(0.6, 0.05, 0.35), drop=True, energy=0.95),
            _region(tempo=128, bands=(0.35, 0.05, 0.6), drop=True, energy=0.95),
        )
        self.assertEqual(match.blend_type, 'double_drop')

    def test_bass_swap_detected(self):
        match = overlay_score(
            _region(tempo=124, bands=(0.6, 0.05, 0.35)),
            _region(tempo=124, bands=(0.6, 0.35, 0.05)),
        )
        self.assertEqual(match.blend_type, 'bass_swap')


class SearchTests(unittest.TestCase):
    def test_best_overlay_picks_complementary_entry(self):
        exits = [_region('A', position=0.9, bands=(0.8, 0.15, 0.05))]
        entries = [
            _region('B', position=0.1, bands=(0.8, 0.15, 0.05)),  # collides (same bass)
            _region('B', position=0.2, bands=(0.05, 0.15, 0.8)),  # complements
        ]
        match = best_overlay(exits, entries)
        self.assertIsNotNone(match)
        np.testing.assert_allclose(match.entry_region.band_profile, [0.05, 0.15, 0.8])

    def test_plan_mix_pairs_consecutive(self):
        regions = {t: [_region(t, position=0.1), _region(t, position=0.9)] for t in ('A', 'B', 'C')}
        plan = plan_mix(regions, ['A', 'B', 'C'])
        self.assertEqual(len(plan), 2)


class NoveltyTests(unittest.TestCase):
    def test_boundary_found_between_two_blocks(self):
        rng = np.random.default_rng(0)
        block1 = rng.normal(0.0, 0.02, size=(30, 6)) + np.array([1, 0, 0, 0, 0, 0])
        block2 = rng.normal(0.0, 0.02, size=(30, 6)) + np.array([0, 0, 0, 0, 0, 1])
        sequence = np.vstack([block1, block2])
        boundaries = foote_novelty_boundaries(sequence, kernel_size=6, top_n=3)
        self.assertTrue(any(24 <= b <= 36 for b in boundaries), boundaries)


if __name__ == '__main__':
    unittest.main()
