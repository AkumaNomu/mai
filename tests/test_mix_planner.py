import os
import tempfile
import unittest

import pandas as pd

from mai.mix_planner import (
    build_regions,
    export_cue_sheet,
    mix_plan_to_dataframe,
    plan_mix_from_dataframe,
    regions_from_row,
)


def _playlist() -> pd.DataFrame:
    return pd.DataFrame([
        {'track_id': 'a', 'title': 'Opener', 'tempo': 122.0, 'key': 5, 'energy': 0.6,
         'spectral_centroid': 2200.0, 'speechiness': 0.05, 'harmonic_ratio': 0.7, 'onset_strength': 2.0},
        {'track_id': 'b', 'title': 'Builder', 'tempo': 123.0, 'key': 7, 'energy': 0.82,
         'spectral_centroid': 3800.0, 'speechiness': 0.04, 'harmonic_ratio': 0.4, 'onset_strength': 3.2},
        {'track_id': 'c', 'title': 'Closer', 'tempo': 121.0, 'key': 5, 'energy': 0.5,
         'spectral_centroid': 1600.0, 'speechiness': 0.06, 'harmonic_ratio': 0.8, 'onset_strength': 1.5},
    ])


class MixPlannerTests(unittest.TestCase):
    def test_regions_from_row_three_sections(self):
        row = _playlist().iloc[1]
        regions = regions_from_row(row, 'b')
        self.assertEqual(len(regions), 3)
        positions = [r.position for r in regions]
        self.assertEqual(positions, sorted(positions))
        for region in regions:
            self.assertAlmostEqual(float(region.band_profile.sum()), 1.0, places=6)
        self.assertTrue(regions[1].is_drop)  # body energy 0.82 > 0.8

    def test_build_regions_keyed_by_id(self):
        regions = build_regions(_playlist())
        self.assertEqual(set(regions), {'a', 'b', 'c'})
        self.assertEqual(len(regions['a']), 3)

    def test_plan_and_cue_sheet(self):
        df = _playlist()
        plan = plan_mix_from_dataframe(df)
        self.assertEqual(len(plan), 2)  # 3 tracks -> 2 transitions
        table = mix_plan_to_dataframe(plan, df)
        self.assertEqual(list(table['from_track']), ['Opener', 'Builder'])
        self.assertEqual(list(table['to_track']), ['Builder', 'Closer'])
        for blend in table['blend_type']:
            self.assertIn(blend, ('long_blend', 'bass_swap', 'double_drop', 'echo_out', 'loop_roll', 'cut'))

    def test_export_cue_sheet_writes_csv(self):
        df = _playlist()
        plan = plan_mix_from_dataframe(df)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'cue.csv')
            export_cue_sheet(plan, path, df)
            reloaded = pd.read_csv(path)
        self.assertEqual(len(reloaded), 2)
        self.assertIn('blend_type', reloaded.columns)


if __name__ == '__main__':
    unittest.main()
