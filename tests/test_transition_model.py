import tempfile
import unittest

import numpy as np
import pandas as pd

from mai.playlist_generation import compute_transition_scores
from mai.transition_model import (
    load_transition_model,
    save_transition_model,
    score_transition_matrix,
    train_transition_model,
)


def _training_rows() -> pd.DataFrame:
    return pd.DataFrame([
        {
            'video_id': 'mix-a',
            'label': 'excellent',
            'label_source': 'curation-a',
            'from_tempo': 120.0,
            'to_tempo': 121.5,
            'from_key': 5,
            'to_key': 5,
            'from_mode': 1,
            'to_mode': 1,
            'from_intro_tempo': 118.0,
            'to_intro_tempo': 120.0,
            'from_outro_tempo': 122.0,
            'to_outro_tempo': 123.0,
            'from_intro_attack_time_s': 1.0,
            'to_intro_attack_time_s': 1.1,
            'from_outro_abruptness': 0.20,
            'to_outro_abruptness': 0.24,
            'from_sentiment_valence': 0.80,
            'to_sentiment_valence': 0.82,
            'from_resolved_title': 'Track A',
            'to_resolved_title': 'Track B',
            'from_resolved_artist': 'Artist A',
            'to_resolved_artist': 'Artist A',
            'from_genre_primary': 'house',
            'to_genre_primary': 'house',
            'from_mix_group': 'house',
            'to_mix_group': 'house',
        },
        {
            'video_id': 'mix-a',
            'label': 'excellent',
            'label_source': 'curation-a',
            'from_tempo': 121.5,
            'to_tempo': 123.0,
            'from_key': 5,
            'to_key': 6,
            'from_mode': 1,
            'to_mode': 1,
            'from_intro_tempo': 120.0,
            'to_intro_tempo': 121.0,
            'from_outro_tempo': 123.0,
            'to_outro_tempo': 124.0,
            'from_intro_attack_time_s': 1.1,
            'to_intro_attack_time_s': 1.2,
            'from_outro_abruptness': 0.24,
            'to_outro_abruptness': 0.25,
            'from_sentiment_valence': 0.82,
            'to_sentiment_valence': 0.84,
            'from_resolved_title': 'Track B',
            'to_resolved_title': 'Track C',
            'from_resolved_artist': 'Artist A',
            'to_resolved_artist': 'Artist A',
            'from_genre_primary': 'house',
            'to_genre_primary': 'house',
            'from_mix_group': 'house',
            'to_mix_group': 'house',
        },
        {
            'video_id': 'mix-b',
            'label': 'excellent',
            'label_source': 'curation-b',
            'from_tempo': 88.0,
            'to_tempo': 90.0,
            'from_key': 1,
            'to_key': 2,
            'from_mode': 0,
            'to_mode': 1,
            'from_intro_tempo': 86.0,
            'to_intro_tempo': 88.0,
            'from_outro_tempo': 90.0,
            'to_outro_tempo': 91.0,
            'from_intro_attack_time_s': 2.4,
            'to_intro_attack_time_s': 2.5,
            'from_outro_abruptness': 0.70,
            'to_outro_abruptness': 0.72,
            'from_sentiment_valence': 0.24,
            'to_sentiment_valence': 0.26,
            'from_resolved_title': 'Track X',
            'to_resolved_title': 'Track Y',
            'from_resolved_artist': 'Artist B',
            'to_resolved_artist': 'Artist C',
            'from_genre_primary': 'ambient',
            'to_genre_primary': 'ambient',
            'from_mix_group': 'ambient',
            'to_mix_group': 'ambient',
        },
        {
            'video_id': 'mix-c',
            'label': 'excellent',
            'label_source': 'curation-c',
            'from_tempo': 91.0,
            'to_tempo': 93.0,
            'from_key': 2,
            'to_key': 3,
            'from_mode': 1,
            'to_mode': 1,
            'from_intro_tempo': 89.0,
            'to_intro_tempo': 91.0,
            'from_outro_tempo': 93.0,
            'to_outro_tempo': 94.0,
            'from_intro_attack_time_s': 2.5,
            'to_intro_attack_time_s': 2.6,
            'from_outro_abruptness': 0.68,
            'to_outro_abruptness': 0.69,
            'from_sentiment_valence': 0.28,
            'to_sentiment_valence': 0.30,
            'from_resolved_title': 'Track Y',
            'to_resolved_title': 'Track Z',
            'from_resolved_artist': 'Artist C',
            'to_resolved_artist': 'Artist D',
            'from_genre_primary': 'ambient',
            'to_genre_primary': 'ambient',
            'from_mix_group': 'ambient',
            'to_mix_group': 'ambient',
        },
    ])


def _playlist_tracks() -> pd.DataFrame:
    return pd.DataFrame([
        {
            'title': 'Track A',
            'artist': 'Artist A',
            'tempo': 120.0,
            'key': 5,
            'mode': 1,
            'intro_tempo': 118.0,
            'outro_tempo': 122.0,
            'intro_attack_time_s': 1.0,
            'outro_abruptness': 0.20,
            'sentiment_valence': 0.81,
            'genre_primary': 'house',
            'mix_group': 'house',
        },
        {
            'title': 'Track B',
            'artist': 'Artist A',
            'tempo': 121.0,
            'key': 5,
            'mode': 1,
            'intro_tempo': 119.0,
            'outro_tempo': 123.0,
            'intro_attack_time_s': 1.1,
            'outro_abruptness': 0.22,
            'sentiment_valence': 0.83,
            'genre_primary': 'house',
            'mix_group': 'house',
        },
        {
            'title': 'Track Z',
            'artist': 'Artist D',
            'tempo': 90.0,
            'key': 1,
            'mode': 0,
            'intro_tempo': 88.0,
            'outro_tempo': 92.0,
            'intro_attack_time_s': 2.6,
            'outro_abruptness': 0.71,
            'sentiment_valence': 0.27,
            'genre_primary': 'ambient',
            'mix_group': 'ambient',
        },
    ])


class TransitionModelTests(unittest.TestCase):
    def test_train_transition_model_builds_a_scored_artifact(self):
        model = train_transition_model(_training_rows(), negative_ratio=1.0, random_state=7, device='cpu')
        scores = score_transition_matrix(model, _playlist_tracks())

        self.assertGreater(model.training_summary['positive_rows'], 0)
        self.assertGreater(model.training_summary['negative_rows'], 0)
        self.assertGreater(model.training_summary['feature_count'], 0)
        self.assertEqual(scores.shape, (3, 3))
        self.assertTrue(np.all(scores >= 0.0))
        self.assertTrue(np.all(scores <= 1.0))
        self.assertAlmostEqual(float(np.diag(scores).sum()), 0.0, places=6)
        self.assertGreater(float(scores[0, 1]), float(scores[0, 2]))

    def test_transition_model_round_trips_through_joblib(self):
        model = train_transition_model(_training_rows(), negative_ratio=1.0, random_state=11, device='cpu')

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = f'{tmpdir}/transition_model.joblib'
            save_transition_model(model, model_path)
            loaded = load_transition_model(model_path)

        original_scores = score_transition_matrix(model, _playlist_tracks())
        loaded_scores = score_transition_matrix(loaded, _playlist_tracks())

        np.testing.assert_allclose(original_scores, loaded_scores, rtol=1e-6, atol=1e-6)

    def test_compute_transition_scores_accepts_transition_model_component(self):
        model = train_transition_model(_training_rows(), negative_ratio=1.0, random_state=13, device='cpu')
        transitions, _ = compute_transition_scores(
            _playlist_tracks(),
            transition_model=model,
            transition_model_weight=0.35,
        )

        self.assertEqual(transitions.shape, (3, 3))
        self.assertTrue(np.all(transitions >= 0.0))
        self.assertTrue(np.all(transitions <= 1.0))


if __name__ == '__main__':
    unittest.main()
