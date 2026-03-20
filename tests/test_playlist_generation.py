import unittest

import numpy as np
import pandas as pd

from mai.playlist_generation import (
    build_transition_report,
    compute_transition_scores,
    generate_playlist_paths,
    ordered_playlist_paths_from_dataframe,
    playlists_to_dataframe,
    transition_score_rating,
)


class PlaylistOrderingTests(unittest.TestCase):
    def test_ordered_playlist_paths_respects_numeric_order_column(self):
        df = pd.DataFrame({
            'title': ['Track A', 'Track B', 'Track C'],
            'sort_order': ['10', '2', '1'],
        })

        paths = ordered_playlist_paths_from_dataframe(df, order_column='sort_order')

        self.assertEqual(paths, [('input_order', [2, 1, 0])])

    def test_ordered_playlist_paths_groups_existing_playlists(self):
        df = pd.DataFrame({
            'playlist_index': [2, 1, 1, 2],
            'position': [2, 2, 1, 1],
            'title': ['D', 'B', 'A', 'C'],
        })

        paths = ordered_playlist_paths_from_dataframe(df)

        self.assertEqual(paths, [('1', [2, 1]), ('2', [3, 0])])


class PlaylistTransitionTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'title': ['Track A', 'Track B', 'Track C'],
            'artist': ['Artist 1', 'Artist 2', 'Artist 3'],
        })
        self.transition_scores = np.asarray(
            [
                [0.0, 0.55, 0.85],
                [0.20, 0.0, 0.70],
                [0.90, 0.40, 0.0],
            ],
            dtype=np.float32,
        )

    def test_playlists_to_dataframe_adds_transition_ratings(self):
        playlist_df = playlists_to_dataframe(
            self.df,
            paths=[[0, 2, 1]],
            transition_scores=self.transition_scores,
        )

        self.assertTrue(pd.isna(playlist_df.loc[0, 'transition_score_from_previous']))
        self.assertEqual(playlist_df.loc[0, 'transition_rating_from_previous'], 'start')
        self.assertAlmostEqual(float(playlist_df.loc[1, 'transition_score_from_previous']), 0.85, places=6)
        self.assertEqual(playlist_df.loc[1, 'transition_rating_from_previous'], 'excellent')
        self.assertAlmostEqual(float(playlist_df.loc[2, 'transition_score_from_previous']), 0.40, places=6)
        self.assertEqual(playlist_df.loc[2, 'transition_rating_from_previous'], 'mixed')

    def test_build_transition_report_flags_suboptimal_next_song(self):
        report = build_transition_report(
            self.df,
            self.transition_scores,
            paths=[[0, 1, 2]],
            playlist_labels=['recommended'],
            report_name='recommended_order',
        )

        first_transition = report.iloc[0]
        self.assertEqual(first_transition['from_label'], 'Artist 1 - Track A')
        self.assertEqual(first_transition['to_label'], 'Artist 2 - Track B')
        self.assertEqual(first_transition['transition_rating'], 'good')
        self.assertEqual(int(first_transition['transition_rank_from_source']), 2)
        self.assertEqual(first_transition['best_possible_next_label'], 'Artist 3 - Track C')
        self.assertAlmostEqual(float(first_transition['best_possible_next_score']), 0.85, places=6)
        self.assertAlmostEqual(float(first_transition['score_gap_to_best']), 0.30, places=6)

    def test_transition_score_rating_bands(self):
        self.assertEqual(transition_score_rating(None), 'start')
        self.assertEqual(transition_score_rating(0.82), 'excellent')
        self.assertEqual(transition_score_rating(0.67), 'strong')
        self.assertEqual(transition_score_rating(0.52), 'good')
        self.assertEqual(transition_score_rating(0.40), 'mixed')
        self.assertEqual(transition_score_rating(0.12), 'rough')

    def test_compute_transition_scores_reports_progress(self):
        df = pd.DataFrame({
            'title': ['Track A', 'Track B'],
            'artist': ['Artist 1', 'Artist 2'],
            'outro_key': [0, 7],
            'outro_mode': [1, 1],
            'intro_key': [0, 7],
            'intro_mode': [1, 1],
        })
        events = []

        transition_scores, _ = compute_transition_scores(
            df,
            progress_callback=lambda label, current, total, detail='': events.append((label, current, total, detail)),
        )

        self.assertEqual(transition_scores.shape, (2, 2))
        self.assertTrue(events)
        self.assertEqual(events[0], ('Transition scoring', 1, 8, 'prepared sentiment features'))
        self.assertEqual(events[-1], ('Transition scoring', 8, 8, 'combined transition matrix'))

    def test_generate_playlist_paths_reports_progress(self):
        df = pd.DataFrame({
            'title': ['A', 'B', 'C', 'D'],
            'artist': ['Artist 1', 'Artist 2', 'Artist 3', 'Artist 4'],
            'style_cluster': ['cluster_1', 'cluster_2', 'cluster_1', 'cluster_2'],
            'genre_primary': ['pop', 'rock', 'pop', 'rock'],
            'genre_confidence': [0.9, 0.9, 0.9, 0.9],
            'genre_source': ['test', 'test', 'test', 'test'],
            'mix_group': ['pop', 'rock', 'pop', 'rock'],
        })
        transition_scores = np.asarray(
            [
                [0.0, 0.8, 0.4, 0.6],
                [0.7, 0.0, 0.9, 0.5],
                [0.6, 0.7, 0.0, 0.8],
                [0.5, 0.6, 0.9, 0.0],
            ],
            dtype=np.float32,
        )
        events = []

        paths, _, resolved_column = generate_playlist_paths(
            df,
            transition_scores,
            playlist_size=2,
            num_playlists=2,
            beam_width=2,
            candidate_width=2,
            progress_callback=lambda label, current, total, detail='': events.append((label, current, total, detail)),
        )

        self.assertEqual(len(paths), 2)
        self.assertEqual(resolved_column, 'mix_group')
        self.assertTrue(events)
        self.assertEqual(events[0], ('Playlist 1/2', 0, 2, 'starting search'))
        self.assertEqual(events[-1][0], 'Playlist 2/2')
        self.assertEqual(events[-1][1], 2)
        self.assertEqual(events[-1][2], 2)


if __name__ == '__main__':
    unittest.main()
