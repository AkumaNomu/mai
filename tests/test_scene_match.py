import unittest

import numpy as np
import pandas as pd

from mai.scene_context import analyze_scene_text
from mai.scene_match import build_scene_target, score_library_against_scene

try:
    from PIL import Image
    from mai.scene_features import analyze_scene_image
    _HAS_PIL = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_PIL = False


def _library() -> pd.DataFrame:
    # Two clearly separated tracks: a bright, major, high-valence "happy" song and
    # a dark, minor, rough "tense" song. Raw descriptors drive the sentiment axes.
    return pd.DataFrame([
        {
            'title': 'Sunshine', 'genre_primary': 'pop',
            'valence': 0.92, 'energy': 0.80, 'danceability': 0.85, 'mode': 1,
            'tempo': 124.0, 'spectral_centroid': 4200.0, 'spectral_flatness': 0.05,
            'zcr': 0.04, 'onset_strength': 3.5, 'harmonic_ratio': 0.85, 'acousticness': 0.2,
            'speechiness': 0.05,
        },
        {
            'title': 'Cold Dread', 'genre_primary': 'industrial',
            'valence': 0.08, 'energy': 0.55, 'danceability': 0.25, 'mode': 0,
            'tempo': 96.0, 'spectral_centroid': 1300.0, 'spectral_flatness': 0.42,
            'zcr': 0.17, 'onset_strength': 2.2, 'harmonic_ratio': 0.20, 'acousticness': 0.1,
            'speechiness': 0.04,
        },
    ])


class SceneContextTests(unittest.TestCase):
    def test_lexicon_maps_horror_to_high_tension_low_valence(self):
        context = analyze_scene_text('a tense horror scene, dark and eerie')
        self.assertIsNotNone(context.mood)
        self.assertGreater(context.mood['tension'], 0.7)
        self.assertLess(context.mood['valence'], 0.4)
        self.assertIn('industrial', context.genre_weights)

    def test_lexicon_maps_romance_to_high_warmth(self):
        context = analyze_scene_text('a tender romantic moment at sunset')
        self.assertIsNotNone(context.mood)
        self.assertGreater(context.mood['warmth'], 0.7)
        self.assertGreater(context.mood['valence'], 0.6)

    def test_no_cue_returns_no_mood(self):
        self.assertIsNone(analyze_scene_text('asdf qwerty 12345').mood)


class SceneMatchTests(unittest.TestCase):
    def test_dark_scene_prefers_tense_track(self):
        target = build_scene_target(scene_text='a tense horror scene in the dark')
        matched = score_library_against_scene(_library(), target)
        self.assertEqual(matched.iloc[0]['title'], 'Cold Dread')
        self.assertGreater(matched.iloc[0]['scene_fit'], matched.iloc[1]['scene_fit'])

    def test_happy_scene_prefers_bright_track(self):
        target = build_scene_target(scene_text='a joyful celebration, hopeful and bright')
        matched = score_library_against_scene(_library(), target)
        self.assertEqual(matched.iloc[0]['title'], 'Sunshine')

    def test_scores_are_bounded_and_ranked(self):
        target = build_scene_target(scene_text='a peaceful calm morning')
        matched = score_library_against_scene(_library(), target, top_k=2)
        fits = matched['scene_fit'].to_numpy()
        self.assertTrue(np.all((fits >= 0.0) & (fits <= 1.0)))
        self.assertTrue(np.all(np.diff(fits) <= 1e-9))  # descending

    def test_requires_some_scene_input(self):
        with self.assertRaises(ValueError):
            build_scene_target()


@unittest.skipUnless(_HAS_PIL, 'Pillow not installed')
class SceneFeaturesTests(unittest.TestCase):
    def _scene(self, rgb):
        import tempfile
        path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
        Image.new('RGB', (48, 48), rgb).save(path)
        return analyze_scene_image(path)

    def test_warm_bright_vs_cold_dark_moods(self):
        warm = self._scene((255, 140, 0))   # vivid orange
        cold = self._scene((12, 14, 60))     # dark navy

        self.assertGreater(warm.warm_share, cold.warm_share)
        self.assertGreater(warm.mood['warmth'], cold.mood['warmth'])
        self.assertGreater(warm.mood['valence'], cold.mood['valence'])
        self.assertLess(warm.dark_share, cold.dark_share)
        self.assertGreater(cold.mood['tension'], warm.mood['tension'])

    def test_mood_values_bounded(self):
        scene = self._scene((128, 64, 200))
        for value in scene.mood.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        self.assertEqual(len(scene.palette), len(scene.palette_weights))


if __name__ == '__main__':
    unittest.main()
