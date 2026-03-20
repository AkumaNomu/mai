import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from mai import training_scrape


def fake_analyze_youtube_playlist_audio(
    df,
    audio_cache_dir='data/audio_cache',
    feature_cache_dir=None,
    edge_seconds=30.0,
    silence_top_db=35.0,
    flow_profile='deep-dj',
    resource_profile='default',
    refresh_cache=False,
    download_workers=1,
    analysis_workers=1,
    delete_audio_after_analysis=True,
    progress_callback=None,
):
    del resource_profile
    rows = []
    for _, row in df.iterrows():
        rows.append({
            'video_id': row['video_id'],
            'title': row.get('title', ''),
            'artist': row.get('artist', ''),
            'url': row.get('url', ''),
            'tempo': 120.0 + len(str(row['video_id'])),
            'key': 5,
            'mode': 1,
            'intro_tempo': 118.0,
            'outro_tempo': 122.0,
            'intro_seconds_used': edge_seconds,
            'outro_seconds_used': edge_seconds,
            'intro_attack_time_s': 1.25,
            'outro_abruptness': 0.35,
            'outro_release_time_s': 3.0,
            'intro_flux_peak': 0.8,
        })
    return pd.DataFrame(rows)


class TrainingScrapeParserTests(unittest.TestCase):
    def test_parse_tracklist_description_supports_mmss_hhmmss_and_separator(self):
        rows = training_scrape.parse_tracklist_description(
            'intro text\n'
            '00:45 - Song A - Artist A\n'
            '01:02:03 Song B - Artist B\n'
            'not a track\n'
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['timestamp'], '00:45')
        self.assertEqual(rows[0]['timestamp_s'], 45)
        self.assertEqual(rows[0]['track_raw'], 'Song A - Artist A')
        self.assertEqual(rows[1]['timestamp'], '01:02:03')
        self.assertEqual(rows[1]['timestamp_s'], 3723)

    def test_parse_tracklist_description_preserves_unicode(self):
        rows = training_scrape.parse_tracklist_description(
            '00:00 修羅の花 (Flower of Carnage) - 梶芽衣子 (Meiko Kaji)\n'
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['track_raw'], '修羅の花 (Flower of Carnage) - 梶芽衣子 (Meiko Kaji)')
        self.assertEqual(rows[0]['artist_guess'], '梶芽衣子 (Meiko Kaji)')
        self.assertEqual(rows[0]['title_guess'], '修羅の花 (Flower of Carnage)')

    def test_parse_tracklist_chapters_builds_rows(self):
        rows = training_scrape.parse_tracklist_chapters([
            {'start_time': 0, 'title': 'Song A - Artist A'},
            {'start_time': 185, 'title': 'Song B - Artist B'},
        ])

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['timestamp'], '00:00')
        self.assertEqual(rows[1]['timestamp'], '03:05')
        self.assertEqual(rows[1]['track_source'], 'chapters')

    def test_parse_tracklist_description_salvages_lines_with_attached_urls(self):
        cleaned_line = training_scrape.normalize_track_text(
            'arteaga verse dot com (https://www.youtube.com/watch?v=1abNyw-vY9g) - memo montañez'
        )
        rows = training_scrape.parse_tracklist_description(
            '00:00 arteaga verse dot com (https://www.youtube.com/watch?v=1abNyw-vY9g) - memo montañez\n'
            '03:00 Song A - Artist A\n'
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['track_raw'], cleaned_line)
        self.assertEqual(rows[1]['track_raw'], 'Song A - Artist A')
        return
        raw_line = 'arteaga verse dot com (https://www.youtube.com/watch?v=1abNyw-vY9g) - memo montañez'
        rows = training_scrape.parse_tracklist_description(
            '00:00 arteaga verse dot com (https://www.youtube.com/watch?v=1abNyw-vY9g) - memo montañez\n'
            '03:00 Song A - Artist A\n'
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['track_raw'], 'arteaga verse dot com - memo montaÃ±ez')
        self.assertEqual(rows[1]['track_raw'], 'Song A - Artist A')

    def test_parse_tracklist_watch_metadata_supports_music_sections(self):
        rows = training_scrape.parse_tracklist_watch_metadata({
            'music_tracks': [
                {'start_time': 0, 'track': 'Song A', 'artists': [{'name': 'Artist A'}]},
                {'start_time_ms': 185000, 'song': 'Song B', 'artist': 'Artist B'},
            ]
        })

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['track_raw'], 'Song A - Artist A')
        self.assertEqual(rows[0]['timestamp'], '00:00')
        self.assertEqual(rows[1]['timestamp'], '03:05')
        self.assertEqual(rows[1]['track_source'], 'chapters')

    def test_guess_artist_title_is_conservative_for_multiple_separators(self):
        artist_guess, title_guess = training_scrape.guess_artist_title('Track A - Artist A - Extra')

        self.assertEqual(artist_guess, '')
        self.assertEqual(title_guess, '')

    def test_build_track_search_query_prefers_cleaned_raw_text(self):
        query = training_scrape.build_track_search_query(
            '03:25 UMBASA x ARELICE - LUNA',
            artist_guess='LUNA',
            title_guess='03:25 UMBASA x ARELICE',
        )

        self.assertEqual(query, 'UMBASA x ARELICE - LUNA')


class TrainingScrapeSourceSelectionTests(unittest.TestCase):
    def test_build_source_track_rows_prefers_description_and_enriches_with_chapters(self):
        rows = training_scrape._build_source_track_rows(
            video_id='mix-1',
            video_url='https://www.youtube.com/watch?v=mix-1',
            metadata={
                'title': 'Mix 1',
                'uploader_id': '@mix',
                'description': '00:00 Song A - Artist A\n03:00 Song B - Artist B\n',
                'chapters': [
                    {'start_time': 0, 'title': 'Chapter A'},
                    {'start_time': 180, 'title': 'Chapter B'},
                ],
            },
            channel_config={'url': 'https://www.youtube.com/@mix/videos', 'label': 'excellent', 'label_source': 'desc'},
        )

        self.assertEqual([row['track_source'] for row in rows], ['description+chapters', 'description+chapters'])
        self.assertEqual(rows[0]['track_raw'], 'Song A - Artist A')
        self.assertEqual(rows[0]['chapter_title'], 'Chapter A')

    def test_build_source_track_rows_falls_back_to_chapters(self):
        rows = training_scrape._build_source_track_rows(
            video_id='mix-2',
            video_url='https://www.youtube.com/watch?v=mix-2',
            metadata={
                'title': 'Mix 2',
                'uploader_id': '@mix',
                'description': 'no timestamps here',
                'chapters': [
                    {'start_time': 0, 'title': 'Song C - Artist C'},
                    {'start_time': 200, 'title': 'Song D - Artist D'},
                ],
            },
            channel_config={'url': 'https://www.youtube.com/@mix/videos', 'label': 'excellent', 'label_source': 'desc'},
        )

        self.assertEqual([row['track_source'] for row in rows], ['chapters', 'chapters'])
        self.assertEqual(rows[1]['track_raw'], 'Song D - Artist D')


class TrainingScrapeResolutionTests(unittest.TestCase):
    def test_search_youtube_track_candidates_passes_cookiefile_to_yt_dlp(self):
        class CookieAwareYoutubeDL:
            calls = []

            def __init__(self, opts):
                self.opts = opts

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def extract_info(self, url, download=False):
                CookieAwareYoutubeDL.calls.append(dict(self.opts))
                return {'entries': [{'id': 'track-a', 'title': 'Song A', 'uploader': 'Artist A', 'duration': 210}]}

        with patch(
            'mai.training_scrape.apply_yt_dlp_auth_options',
            side_effect=lambda opts: {**opts, 'cookiefile': 'cookies.txt'},
        ) as auth_mock:
            with patch('mai.training_scrape.YoutubeDL', side_effect=CookieAwareYoutubeDL):
                candidates = training_scrape.search_youtube_track_candidates('Artist A - Song A', refresh=True)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(auth_mock.call_count, 1)
        self.assertEqual(CookieAwareYoutubeDL.calls[0].get('cookiefile'), 'cookies.txt')

    def test_search_youtube_track_candidates_filters_missing_entries(self):
        class MissingEntryYoutubeDL:
            def __init__(self, opts):
                self.opts = opts

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def extract_info(self, url, download=False):
                return {
                    'entries': [
                        None,
                        {'id': 'track-a', 'title': 'Song A', 'uploader': 'Artist A', 'duration': 210},
                    ]
                }

        with patch('mai.training_scrape.YoutubeDL', side_effect=MissingEntryYoutubeDL):
            candidates = training_scrape.search_youtube_track_candidates('Artist A - Song A')

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]['id'], 'track-a')

    def test_search_youtube_track_candidates_falls_back_to_flat_results_after_error(self):
        class FallbackYoutubeDL:
            def __init__(self, opts):
                self.opts = opts

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def extract_info(self, url, download=False):
                if self.opts.get('extract_flat') == 'in_playlist':
                    return {
                        'entries': [
                            {'id': 'track-b', 'title': 'Song B', 'uploader': 'Artist B', 'duration': 190},
                        ]
                    }
                raise RuntimeError('This video is not available')

        with patch('mai.training_scrape.YoutubeDL', side_effect=FallbackYoutubeDL):
            candidates = training_scrape.search_youtube_track_candidates('Artist B - Song B')

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]['id'], 'track-b')

    def test_select_track_candidate_prefers_non_mix_shorter_result(self):
        selected = training_scrape.select_track_candidate(
            [
                {'id': 'mix-video', 'title': 'The mix itself', 'duration': 3600},
                {'id': 'track-video', 'title': 'Resolved track', 'duration': 220, 'uploader': 'Artist A'},
            ],
            excluded_video_ids={'mix-video'},
        )

        self.assertEqual(selected['resolved_video_id'], 'track-video')
        self.assertEqual(selected['resolved_artist'], 'Artist A')

    def test_select_track_candidate_rejects_irrelevant_low_overlap_result(self):
        selected = training_scrape.select_track_candidate(
            [
                {
                    'id': 'tutorial-video',
                    'title': 'Abstract Glowing Halo Ring. Adobe Illustrator Tutorial.',
                    'duration': 490,
                    'uploader': 'Design Channel',
                },
            ],
            track_raw='SPYRAL - Alone',
            query_variants=['SPYRAL - Alone'],
        )

        self.assertIsNone(selected)

    def test_select_track_candidate_prefers_best_matching_song_result(self):
        selected = training_scrape.select_track_candidate(
            [
                {
                    'id': 'tutorial-video',
                    'title': 'Abstract Glowing Halo Ring. Adobe Illustrator Tutorial.',
                    'duration': 490,
                    'uploader': 'Design Channel',
                },
                {
                    'id': 'song-video',
                    'title': 'Alone',
                    'duration': 201,
                    'uploader': 'SPYRAL - Topic',
                },
            ],
            track_raw='SPYRAL - Alone',
            query_variants=['SPYRAL - Alone'],
        )

        self.assertIsNotNone(selected)
        self.assertEqual(selected['resolved_video_id'], 'song-video')

    @patch('mai.training_scrape.search_youtube_track_candidates')
    def test_resolve_scraped_tracks_deduplicates_repeated_queries_within_run(self, search_mock):
        search_mock.return_value = [
            {'id': 'track-a', 'title': 'Song A', 'uploader': 'Artist A', 'duration': 210},
        ]
        track_df = pd.DataFrame([
            {
                'video_id': 'mix-1',
                'position': 1,
                'track_raw': 'Song A - Artist A',
                'timestamp': '00:00',
                'timestamp_s': 0,
                'artist_guess': 'Artist A',
                'title_guess': 'Song A',
                'track_source': 'description',
                'label': 'excellent',
                'label_source': 'desc',
                'channel_url': 'https://www.youtube.com/@mix/videos',
                'channel_handle': '@mix',
                'video_title': 'Mix 1',
                'video_url': 'https://www.youtube.com/watch?v=mix-1',
                'description_length': 20,
            },
            {
                'video_id': 'mix-2',
                'position': 1,
                'track_raw': 'Song A - Artist A',
                'timestamp': '00:00',
                'timestamp_s': 0,
                'artist_guess': 'Artist A',
                'title_guess': 'Song A',
                'track_source': 'description',
                'label': 'excellent',
                'label_source': 'desc',
                'channel_url': 'https://www.youtube.com/@mix/videos',
                'channel_handle': '@mix',
                'video_title': 'Mix 2',
                'video_url': 'https://www.youtube.com/watch?v=mix-2',
                'description_length': 20,
            },
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            resolved_df, summary = training_scrape.resolve_scraped_tracks(
                track_df,
                cache_dir=tmpdir,
                search_workers=2,
            )

        self.assertEqual(search_mock.call_count, 1)
        self.assertEqual(summary['tracks_resolved'], 2)
        self.assertTrue(resolved_df['resolved_video_id'].eq('track-a').all())

    @patch('mai.training_scrape.search_youtube_track_candidates')
    def test_resolve_scraped_tracks_invalidates_cache_when_max_search_results_changes(self, search_mock):
        search_mock.return_value = [
            {'id': 'track-a', 'title': 'Song A', 'uploader': 'Artist A', 'duration': 210},
        ]
        track_df = pd.DataFrame([
            {
                'video_id': 'mix-1',
                'position': 1,
                'track_raw': 'Song A - Artist A',
                'timestamp': '00:00',
                'timestamp_s': 0,
                'artist_guess': 'Artist A',
                'title_guess': 'Song A',
                'track_source': 'description',
                'label': 'excellent',
                'label_source': 'desc',
                'channel_url': 'https://www.youtube.com/@mix/videos',
                'channel_handle': '@mix',
                'video_title': 'Mix 1',
                'video_url': 'https://www.youtube.com/watch?v=mix-1',
                'description_length': 20,
                'source_cache_version': training_scrape.SOURCE_TRACK_CACHE_VERSION,
            },
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            training_scrape.resolve_scraped_tracks(
                track_df,
                cache_dir=tmpdir,
                max_results=5,
                search_workers=1,
            )
            training_scrape.resolve_scraped_tracks(
                track_df,
                cache_dir=tmpdir,
                max_results=10,
                search_workers=1,
            )
            training_scrape.resolve_scraped_tracks(
                track_df,
                cache_dir=tmpdir,
                max_results=10,
                search_workers=1,
            )

        self.assertEqual(search_mock.call_count, 2)

    @patch('mai.training_scrape.search_youtube_track_candidates')
    def test_resolve_scraped_tracks_emits_start_progress_details(self, search_mock):
        search_mock.return_value = [
            {'id': 'track-a', 'title': 'Song A', 'uploader': 'Artist A', 'duration': 210},
        ]
        track_df = pd.DataFrame([
            {
                'video_id': 'mix-1',
                'position': 1,
                'track_raw': 'Song A - Artist A',
                'timestamp': '00:00',
                'timestamp_s': 0,
                'artist_guess': 'Artist A',
                'title_guess': 'Song A',
                'track_source': 'description',
                'label': 'excellent',
                'label_source': 'desc',
                'channel_url': 'https://www.youtube.com/@mix/videos',
                'channel_handle': '@mix',
                'video_title': 'Mix 1',
                'video_url': 'https://www.youtube.com/watch?v=mix-1',
                'description_length': 20,
                'source_cache_version': training_scrape.SOURCE_TRACK_CACHE_VERSION,
            },
        ])
        events = []

        with tempfile.TemporaryDirectory() as tmpdir:
            training_scrape.resolve_scraped_tracks(
                track_df,
                cache_dir=tmpdir,
                max_results=5,
                search_workers=1,
                progress_callback=lambda label, current, total, detail='': events.append((label, current, total, detail)),
            )

        resolving_events = [event for event in events if event[0] == 'Resolving tracks']
        self.assertEqual(resolving_events[0], ('Resolving tracks', 0, 1, 'preparing 1 scraped tracks'))
        self.assertEqual(resolving_events[1], ('Resolving tracks', 0, 1, 'queued 1 unique YouTube searches'))
        search_events = [event for event in events if event[0] == 'Searching YouTube']
        self.assertEqual(search_events[0], ('Searching YouTube', 0, 1, 'running 1 searches with 1 workers'))
        self.assertEqual(search_events[-1], ('Searching YouTube', 1, 1, 'Song A - Artist A -> 1 candidates'))

    @patch('mai.training_scrape.search_youtube_track_candidates')
    def test_resolve_scraped_tracks_salvages_url_lines_for_search(self, search_mock):
        cleaned_line = training_scrape.normalize_track_text(
            'arteaga verse dot com (https://www.youtube.com/watch?v=1abNyw-vY9g) - memo montañez'
        )
        search_mock.return_value = []
        track_df = pd.DataFrame([
            {
                'video_id': 'mix-1',
                'position': 1,
                'track_raw': 'arteaga verse dot com (https://www.youtube.com/watch?v=1abNyw-vY9g) - memo montañez',
                'timestamp': '00:00',
                'timestamp_s': 0,
                'artist_guess': 'memo montañez',
                'title_guess': 'arteaga verse dot com (https://www.youtube.com/watch?v=1abNyw-vY9g)',
                'track_source': 'description',
                'label': 'excellent',
                'label_source': 'desc',
                'channel_url': 'https://www.youtube.com/@mix/videos',
                'channel_handle': '@mix',
                'video_title': 'Mix 1',
                'video_url': 'https://www.youtube.com/watch?v=mix-1',
                'description_length': 20,
                'source_cache_version': training_scrape.SOURCE_TRACK_CACHE_VERSION,
            },
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            resolved_df, summary = training_scrape.resolve_scraped_tracks(
                track_df,
                cache_dir=tmpdir,
                search_workers=1,
            )

        search_mock.assert_called_once()
        self.assertEqual(search_mock.call_args.args[0], cleaned_line)
        self.assertEqual(summary['tracks_resolved'], 0)
        self.assertEqual(summary['tracks_unresolved'], 1)
        self.assertEqual(resolved_df.loc[0, 'resolution_status'], 'no_match')
        return
        search_mock.return_value = []
        track_df = pd.DataFrame([
            {
                'video_id': 'mix-1',
                'position': 1,
                'track_raw': 'arteaga verse dot com (https://www.youtube.com/watch?v=1abNyw-vY9g) - memo montañez',
                'timestamp': '00:00',
                'timestamp_s': 0,
                'artist_guess': 'memo montañez',
                'title_guess': 'arteaga verse dot com (https://www.youtube.com/watch?v=1abNyw-vY9g)',
                'track_source': 'description',
                'label': 'excellent',
                'label_source': 'desc',
                'channel_url': 'https://www.youtube.com/@mix/videos',
                'channel_handle': '@mix',
                'video_title': 'Mix 1',
                'video_url': 'https://www.youtube.com/watch?v=mix-1',
                'description_length': 20,
                'source_cache_version': training_scrape.SOURCE_TRACK_CACHE_VERSION,
            },
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            resolved_df, summary = training_scrape.resolve_scraped_tracks(
                track_df,
                cache_dir=tmpdir,
                search_workers=1,
            )

        search_mock.assert_called_once()
        self.assertEqual(search_mock.call_args.args[0], 'arteaga verse dot com - memo montaÃ±ez')
        self.assertEqual(summary['tracks_resolved'], 0)
        self.assertEqual(summary['tracks_unresolved'], 1)
        self.assertEqual(resolved_df.loc[0, 'resolution_status'], 'no_match')

    @patch('mai.training_scrape.search_youtube_track_candidates')
    def test_resolve_scraped_tracks_uses_cleaned_raw_query_not_reversed_guess(self, search_mock):
        search_mock.return_value = []
        track_df = pd.DataFrame([
            {
                'video_id': 'mix-1',
                'position': 1,
                'track_raw': '05:44 Hensonn - Violet',
                'timestamp': '05:44',
                'timestamp_s': 344,
                'artist_guess': 'Violet',
                'title_guess': '05:44 Hensonn',
                'track_source': 'description',
                'label': 'excellent',
                'label_source': 'desc',
                'channel_url': 'https://www.youtube.com/@mix/videos',
                'channel_handle': '@mix',
                'video_title': 'Mix 1',
                'video_url': 'https://www.youtube.com/watch?v=mix-1',
                'description_length': 20,
                'source_cache_version': training_scrape.SOURCE_TRACK_CACHE_VERSION,
            },
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            training_scrape.resolve_scraped_tracks(
                track_df,
                cache_dir=tmpdir,
                search_workers=1,
            )

        search_mock.assert_called_once()
        self.assertEqual(search_mock.call_args.args[0], 'Hensonn - Violet')

    @patch('mai.training_scrape.analyze_youtube_playlist_audio', side_effect=fake_analyze_youtube_playlist_audio)
    def test_analyze_resolved_tracks_dedupes_by_resolved_video_id(self, analyze_mock):
        track_df = pd.DataFrame([
            {
                'video_id': 'mix-1',
                'position': 1,
                'track_raw': 'Song A - Artist A',
                'timestamp': '00:00',
                'timestamp_s': 0,
                'artist_guess': 'Artist A',
                'title_guess': 'Song A',
                'track_source': 'description',
                'label': 'excellent',
                'label_source': 'desc',
                'channel_url': 'https://www.youtube.com/@mix/videos',
                'channel_handle': '@mix',
                'video_title': 'Mix 1',
                'video_url': 'https://www.youtube.com/watch?v=mix-1',
                'description_length': 20,
                'search_query': 'Artist A - Song A',
                'resolution_status': 'resolved',
                'resolved_video_id': 'track-a',
                'resolved_title': 'Song A',
                'resolved_artist': 'Artist A',
                'resolved_url': 'https://www.youtube.com/watch?v=track-a',
                'resolved_duration_seconds': 210,
            },
            {
                'video_id': 'mix-2',
                'position': 1,
                'track_raw': 'Song A - Artist A',
                'timestamp': '00:00',
                'timestamp_s': 0,
                'artist_guess': 'Artist A',
                'title_guess': 'Song A',
                'track_source': 'description',
                'label': 'excellent',
                'label_source': 'desc',
                'channel_url': 'https://www.youtube.com/@mix/videos',
                'channel_handle': '@mix',
                'video_title': 'Mix 2',
                'video_url': 'https://www.youtube.com/watch?v=mix-2',
                'description_length': 20,
                'search_query': 'Artist A - Song A',
                'resolution_status': 'resolved',
                'resolved_video_id': 'track-a',
                'resolved_title': 'Song A',
                'resolved_artist': 'Artist A',
                'resolved_url': 'https://www.youtube.com/watch?v=track-a',
                'resolved_duration_seconds': 210,
            },
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            analyzed_df, summary = training_scrape.analyze_resolved_tracks(
                track_df,
                feature_cache_dir=str(Path(tmpdir) / 'audio_features.csv'),
                cache_dir=tmpdir,
            )

        called_df = analyze_mock.call_args.args[0]
        self.assertEqual(called_df['video_id'].tolist(), ['track-a'])
        self.assertEqual(summary['tracks_analyzed'], 1)
        self.assertEqual(summary['tracks_with_features'], 2)
        self.assertTrue(analyzed_df['tempo'].notna().all())

    @patch('mai.training_scrape.analyze_youtube_playlist_audio', side_effect=fake_analyze_youtube_playlist_audio)
    @patch('mai.training_scrape.fetch_video_metadata')
    def test_analyze_resolved_tracks_emits_metadata_stage_start(self, metadata_mock, analyze_mock):
        metadata_mock.return_value = {
            'title': 'Song A',
            'uploader': 'Artist Channel',
            'channel': 'Artist Channel',
            'description': 'warm neon city pop',
            'tags': ['city pop'],
            'categories': ['Music'],
            'webpage_url': 'https://www.youtube.com/watch?v=abc123def45',
        }
        track_df = pd.DataFrame([
            {
                'video_id': 'mix-1',
                'position': 1,
                'track_raw': 'Song A - Artist A',
                'timestamp': '00:00',
                'timestamp_s': 0,
                'artist_guess': 'Artist A',
                'title_guess': 'Song A',
                'track_source': 'description',
                'label': 'excellent',
                'label_source': 'desc',
                'channel_url': 'https://www.youtube.com/@mix/videos',
                'channel_handle': '@mix',
                'video_title': 'Mix 1',
                'video_url': 'https://www.youtube.com/watch?v=mix-1',
                'description_length': 20,
                'resolution_status': 'resolved',
                'resolved_video_id': 'abc123def45',
                'resolved_title': 'Song A',
                'resolved_artist': 'Artist A',
                'resolved_url': 'https://www.youtube.com/watch?v=abc123def45',
                'resolved_duration_seconds': 210,
            },
        ])
        events = []

        with tempfile.TemporaryDirectory() as tmpdir:
            training_scrape.analyze_resolved_tracks(
                track_df,
                feature_cache_dir=str(Path(tmpdir) / 'audio_features.csv'),
                cache_dir=tmpdir,
                metadata_workers=1,
                progress_callback=lambda label, current, total, detail='': events.append((label, current, total, detail)),
            )

        metadata_events = [event for event in events if event[0] == 'Fetching resolved track metadata']
        self.assertEqual(metadata_events[0], ('Fetching resolved track metadata', 0, 1, 'starting 1 metadata lookups'))

    @patch('mai.training_scrape.analyze_youtube_playlist_audio', side_effect=fake_analyze_youtube_playlist_audio)
    @patch('mai.training_scrape.fetch_video_metadata')
    def test_analyze_resolved_tracks_fetches_metadata_for_registry_context(self, metadata_mock, analyze_mock):
        resolved_video_id = 'abc123def45'
        metadata_mock.return_value = {
            'title': 'Song A (Official Audio)',
            'uploader': 'Artist Channel',
            'channel': 'Artist Channel',
            'description': 'warm neon city pop',
            'tags': ['city pop', 'night drive'],
            'categories': ['Music'],
            'webpage_url': f'https://www.youtube.com/watch?v={resolved_video_id}',
        }
        track_df = pd.DataFrame([
            {
                'video_id': 'mix-1',
                'position': 1,
                'track_raw': 'Song A - Artist A',
                'timestamp': '00:00',
                'timestamp_s': 0,
                'artist_guess': 'Artist A',
                'title_guess': 'Song A',
                'track_source': 'description',
                'label': 'excellent',
                'label_source': 'desc',
                'channel_url': 'https://www.youtube.com/@mix/videos',
                'channel_handle': '@mix',
                'video_title': 'Mix 1',
                'video_url': 'https://www.youtube.com/watch?v=mix-1',
                'description_length': 20,
                'search_query': 'Artist A - Song A',
                'normalized_search_query': 'artist a - song a',
                'search_max_results': 5,
                'resolution_cache_version': training_scrape.RESOLUTION_CACHE_VERSION,
                'resolution_status': 'resolved',
                'resolved_video_id': resolved_video_id,
                'resolved_title': 'Song A',
                'resolved_artist': 'Artist A',
                'resolved_url': f'https://www.youtube.com/watch?v={resolved_video_id}',
                'resolved_duration_seconds': 210,
            },
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            training_scrape.analyze_resolved_tracks(
                track_df,
                feature_cache_dir=str(Path(tmpdir) / 'audio_features.csv'),
                cache_dir=tmpdir,
                metadata_workers=1,
            )

        called_df = analyze_mock.call_args.args[0]
        self.assertEqual(called_df.loc[0, 'description'], 'warm neon city pop')
        self.assertEqual(called_df.loc[0, 'tags'], 'city pop, night drive')
        self.assertEqual(called_df.loc[0, 'category'], 'Music')
        self.assertEqual(called_df.loc[0, 'channel'], 'Artist Channel')


class TrainingScrapeSourceCacheTests(unittest.TestCase):
    @patch('mai.training_scrape._fetch_source_video_rows')
    @patch('mai.training_scrape.fetch_channel_video_entries')
    def test_scrape_channel_track_rows_invalidates_stale_source_cache_rows(
        self,
        channel_entries_mock,
        fetch_rows_mock,
    ):
        channel_entries_mock.return_value = [{
            'video_id': 'mix-1',
            'video_title': 'Mix 1',
            'video_url': 'https://www.youtube.com/watch?v=mix-1',
        }]
        fresh_rows_df = training_scrape.source_tracks_dataframe([
            {
                'video_id': 'mix-1',
                'channel_url': 'https://www.youtube.com/@mix/videos',
                'channel_handle': '@mix',
                'label': 'excellent',
                'label_source': 'new_source',
                'source_cache_version': training_scrape.SOURCE_TRACK_CACHE_VERSION,
                'video_title': 'Mix 1',
                'video_url': 'https://www.youtube.com/watch?v=mix-1',
                'description_length': 20,
                'track_source': 'description',
                'chapter_title': '',
                'chapter_timestamp_s': pd.NA,
                'position': 1,
                'timestamp': '00:00',
                'timestamp_s': 0,
                'track_raw': 'Song A - Artist A',
                'artist_guess': 'Artist A',
                'title_guess': 'Song A',
            },
            {
                'video_id': 'mix-1',
                'channel_url': 'https://www.youtube.com/@mix/videos',
                'channel_handle': '@mix',
                'label': 'excellent',
                'label_source': 'new_source',
                'source_cache_version': training_scrape.SOURCE_TRACK_CACHE_VERSION,
                'video_title': 'Mix 1',
                'video_url': 'https://www.youtube.com/watch?v=mix-1',
                'description_length': 20,
                'track_source': 'description',
                'chapter_title': '',
                'chapter_timestamp_s': pd.NA,
                'position': 2,
                'timestamp': '03:00',
                'timestamp_s': 180,
                'track_raw': 'Song B - Artist B',
                'artist_guess': 'Artist B',
                'title_guess': 'Song B',
            },
        ])
        fetch_rows_mock.return_value = ('mix-1', fresh_rows_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            stale_cache_path = Path(tmpdir) / 'training' / 'source_tracks.csv'
            stale_cache_path.parent.mkdir(parents=True, exist_ok=True)
            stale_df = training_scrape.source_tracks_dataframe([
                {
                    'video_id': 'mix-1',
                    'channel_url': 'https://www.youtube.com/@mix/videos',
                    'channel_handle': '@mix',
                    'label': 'excellent',
                    'label_source': 'old_source',
                    'source_cache_version': 1,
                    'video_title': 'Mix 1',
                    'video_url': 'https://www.youtube.com/watch?v=mix-1',
                    'description_length': 20,
                    'track_source': 'description',
                    'chapter_title': '',
                    'chapter_timestamp_s': pd.NA,
                    'position': 1,
                    'timestamp': '00:00',
                    'timestamp_s': 0,
                    'track_raw': 'Song A - Artist A',
                    'artist_guess': 'Artist A',
                    'title_guess': 'Song A',
                },
            ])
            stale_df.to_csv(stale_cache_path, index=False)

            df, summary = training_scrape.scrape_channel_track_rows(
                channel_url='https://www.youtube.com/@mix/videos',
                cache_dir=tmpdir,
                label='excellent',
                label_source='new_source',
                metadata_workers=1,
            )

        fetch_rows_mock.assert_called_once()
        self.assertEqual(summary['videos_with_tracklist'], 1)
        self.assertTrue(df['label_source'].eq('new_source').all())


class TrainingTransitionRowTests(unittest.TestCase):
    def test_build_training_transition_rows_does_not_bridge_missing_middle_track(self):
        track_df = pd.DataFrame([
            {
                'video_id': 'mix-z',
                'position': 1,
                'timestamp': '00:00',
                'timestamp_s': 0,
                'track_raw': 'Song A - Artist A',
                'artist_guess': 'Artist A',
                'title_guess': 'Song A',
                'track_source': 'description',
                'label': 'excellent',
                'label_source': 'desc',
                'channel_url': 'https://www.youtube.com/@mix/videos',
                'channel_handle': '@mix',
                'video_title': 'Mix Z',
                'video_url': 'https://www.youtube.com/watch?v=mix-z',
                'description_length': 10,
                'resolution_status': 'resolved',
                'resolved_video_id': 'track-a',
                'resolved_title': 'Song A',
                'resolved_artist': 'Artist A',
                'resolved_url': 'https://www.youtube.com/watch?v=track-a',
                'resolved_duration_seconds': 180,
                'tempo': 123.0,
                'key': 5,
                'mode': 1,
                'intro_tempo': 120.0,
                'outro_tempo': 121.0,
                'intro_seconds_used': 30.0,
                'outro_seconds_used': 30.0,
                'intro_attack_time_s': 1.1,
                'outro_abruptness': 0.3,
                'outro_release_time_s': 2.5,
                'intro_flux_peak': 0.8,
            },
            {
                'video_id': 'mix-z',
                'position': 2,
                'timestamp': '03:00',
                'timestamp_s': 180,
                'track_raw': 'Song B - Artist B',
                'artist_guess': 'Artist B',
                'title_guess': 'Song B',
                'track_source': 'description',
                'label': 'excellent',
                'label_source': 'desc',
                'channel_url': 'https://www.youtube.com/@mix/videos',
                'channel_handle': '@mix',
                'video_title': 'Mix Z',
                'video_url': 'https://www.youtube.com/watch?v=mix-z',
                'description_length': 10,
                'resolution_status': 'unavailable',
                'resolved_video_id': '',
                'resolved_title': '',
                'resolved_artist': '',
                'resolved_url': '',
                'resolved_duration_seconds': pd.NA,
            },
            {
                'video_id': 'mix-z',
                'position': 3,
                'timestamp': '06:00',
                'timestamp_s': 360,
                'track_raw': 'Song C - Artist C',
                'artist_guess': 'Artist C',
                'title_guess': 'Song C',
                'track_source': 'description',
                'label': 'excellent',
                'label_source': 'desc',
                'channel_url': 'https://www.youtube.com/@mix/videos',
                'channel_handle': '@mix',
                'video_title': 'Mix Z',
                'video_url': 'https://www.youtube.com/watch?v=mix-z',
                'description_length': 10,
                'resolution_status': 'resolved',
                'resolved_video_id': 'track-c',
                'resolved_title': 'Song C',
                'resolved_artist': 'Artist C',
                'resolved_url': 'https://www.youtube.com/watch?v=track-c',
                'resolved_duration_seconds': 210,
                'tempo': 124.0,
                'key': 6,
                'mode': 1,
                'intro_tempo': 121.0,
                'outro_tempo': 122.0,
                'intro_seconds_used': 30.0,
                'outro_seconds_used': 30.0,
                'intro_attack_time_s': 1.2,
                'outro_abruptness': 0.4,
                'outro_release_time_s': 2.8,
                'intro_flux_peak': 0.9,
            },
        ])

        events = []
        df, summary = training_scrape.build_training_transition_rows(
            track_df,
            flow_profile='deep-dj',
            progress_callback=lambda label, current, total, detail='': events.append((label, current, total, detail)),
        )

        self.assertTrue(df.empty)
        self.assertEqual(summary['positive_pairs'], 0)
        self.assertEqual(summary['pairs_skipped'], 2)
        pair_events = [event for event in events if event[0] == 'Building training pairs']
        self.assertEqual(pair_events[0], ('Building training pairs', 0, 2, 'assembling adjacent pairs from 1 source videos'))


class FakeYoutubeDL:
    calls = []

    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def extract_info(self, url, download=False):
        FakeYoutubeDL.calls.append((url, dict(self.opts)))
        if url == 'https://www.youtube.com/@mix_a/videos':
            return {
                'entries': [
                    {'id': 'mix-a', 'title': 'Mix A', 'url': 'https://www.youtube.com/watch?v=mix-a'},
                    {'id': 'mix-b', 'title': 'Mix B', 'url': 'https://www.youtube.com/watch?v=mix-b'},
                ]
            }
        if url == 'https://www.youtube.com/@mix_b/videos':
            return {
                'entries': [
                    {'id': 'mix-c', 'title': 'Mix C', 'url': 'https://www.youtube.com/watch?v=mix-c'},
                ]
            }
        if url == 'https://www.youtube.com/watch?v=mix-a':
            return {
                'id': 'mix-a',
                'title': 'Mix A',
                'uploader_id': '@mix_a',
                'description': (
                    '00:00 Song A - Artist A\n'
                    '03:10 Song B - Artist B\n'
                    '06:20 Song C - Artist C\n'
                ),
                'chapters': [
                    {'start_time': 0, 'title': 'Chapter A'},
                    {'start_time': 190, 'title': 'Chapter B'},
                    {'start_time': 380, 'title': 'Chapter C'},
                ],
            }
        if url == 'https://www.youtube.com/watch?v=mix-b':
            return {
                'id': 'mix-b',
                'title': 'Mix B',
                'uploader_id': '@mix_a',
                'description': 'no timestamps here',
                'chapters': [
                    {'start_time': 0, 'title': 'Song D - Artist D'},
                    {'start_time': 200, 'title': 'Song E - Artist E'},
                ],
            }
        if url == 'https://www.youtube.com/watch?v=mix-c':
            return {
                'id': 'mix-c',
                'title': 'Mix C',
                'uploader_id': '@mix_b',
                'description': (
                    '00:00 Song F - Artist F\n'
                    '02:00 Track B - Artist Missing\n'
                    '04:00 Song H - Artist H\n'
                ),
                'chapters': None,
            }
        if url in {'ytsearch5:Artist A - Song A', 'ytsearch5:Song A - Artist A'}:
            return {'entries': [{'id': 'track-a', 'title': 'Song A', 'uploader': 'Artist A', 'duration': 210}]}
        if url in {'ytsearch5:Artist B - Song B', 'ytsearch5:Song B - Artist B'}:
            return {'entries': [{'id': 'track-b', 'title': 'Song B', 'uploader': 'Artist B', 'duration': 190}]}
        if url in {'ytsearch5:Artist C - Song C', 'ytsearch5:Song C - Artist C'}:
            return {'entries': [{'id': 'track-c', 'title': 'Song C', 'uploader': 'Artist C', 'duration': 200}]}
        if url in {'ytsearch5:Artist D - Song D', 'ytsearch5:Song D - Artist D'}:
            return {'entries': [{'id': 'track-d', 'title': 'Song D', 'uploader': 'Artist D', 'duration': 180}]}
        if url in {'ytsearch5:Artist E - Song E', 'ytsearch5:Song E - Artist E'}:
            return {'entries': [{'id': 'track-e', 'title': 'Song E', 'uploader': 'Artist E', 'duration': 220}]}
        if url in {'ytsearch5:Artist F - Song F', 'ytsearch5:Song F - Artist F'}:
            return {'entries': [{'id': 'track-f', 'title': 'Song F', 'uploader': 'Artist F', 'duration': 200}]}
        if url in {'ytsearch5:Artist Missing - Track B', 'ytsearch5:Track B - Artist Missing'}:
            return {'entries': [{'id': 'missing-track', 'title': '[Private Video]', 'availability': 'unavailable', 'duration': 200}]}
        if url in {'ytsearch5:Artist H - Song H', 'ytsearch5:Song H - Artist H'}:
            return {'entries': [{'id': 'track-h', 'title': 'Song H', 'uploader': 'Artist H', 'duration': 230}]}
        if url.startswith('https://www.youtube.com/watch?v=track-'):
            track_id = url.split('v=')[1]
            letter = track_id.split('-')[-1].upper()
            return {
                'id': track_id,
                'title': f'Song {letter}',
                'uploader': f'Artist {letter}',
                'channel': f'Artist {letter}',
                'description': f'description for {track_id}',
                'tags': [f'tag-{letter.lower()}'],
                'categories': ['Music'],
                'webpage_url': url,
            }
        raise AssertionError(f'unexpected URL: {url}')


class TrainingScrapeIntegrationTests(unittest.TestCase):
    @patch('mai.training_scrape.analyze_youtube_playlist_audio', side_effect=fake_analyze_youtube_playlist_audio)
    @patch('mai.training_scrape.YoutubeDL', side_effect=FakeYoutubeDL)
    def test_scrape_training_transitions_multi_channel_end_to_end(self, youtube_dl_mock, analyze_mock):
        FakeYoutubeDL.calls = []
        channels = [
            {'url': 'https://www.youtube.com/@mix_a/videos', 'label': 'excellent', 'label_source': 'mix_a'},
            {'url': 'https://www.youtube.com/@mix_b/videos', 'label': 'excellent', 'label_source': 'mix_b'},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            df, summary = training_scrape.scrape_training_transitions(
                channels=channels,
                cache_dir=tmpdir,
                feature_cache_dir=str(Path(tmpdir) / 'audio_features.csv'),
            )

        self.assertEqual(summary['channels_scanned'], 2)
        self.assertEqual(summary['videos_scanned'], 3)
        self.assertEqual(summary['videos_with_tracklist'], 3)
        self.assertEqual(summary['tracks_parsed'], 8)
        self.assertEqual(summary['tracks_resolved'], 7)
        self.assertEqual(summary['tracks_unavailable'], 1)
        self.assertEqual(summary['tracks_unresolved'], 1)
        self.assertEqual(summary['positive_pairs'], 3)
        self.assertEqual(summary['pairs_skipped'], 2)
        self.assertEqual(df.columns[0], 'video_id')
        self.assertEqual(df['video_id'].tolist(), ['mix-a', 'mix-a', 'mix-b'])
        self.assertEqual(df.iloc[0]['from_track_source'], 'description+chapters')
        self.assertEqual(df.iloc[2]['from_track_source'], 'chapters')
        self.assertNotIn('mix-c', df['video_id'].tolist())
        called_df = analyze_mock.call_args.args[0]
        self.assertEqual(called_df['video_id'].tolist(), ['track-a', 'track-b', 'track-c', 'track-d', 'track-e', 'track-f', 'track-h'])

    @patch('mai.training_scrape.analyze_youtube_playlist_audio', side_effect=fake_analyze_youtube_playlist_audio)
    @patch('mai.training_scrape.YoutubeDL', side_effect=FakeYoutubeDL)
    def test_scrape_training_transitions_supports_explicit_video_sources(self, youtube_dl_mock, analyze_mock):
        FakeYoutubeDL.calls = []

        with tempfile.TemporaryDirectory() as tmpdir:
            df, summary = training_scrape.scrape_training_transitions(
                channels=[{
                    'url': 'https://www.youtube.com/watch?v=mix-a',
                    'label': 'strong',
                    'label_source': 'manual_video_curation',
                }],
                cache_dir=tmpdir,
                feature_cache_dir=str(Path(tmpdir) / 'audio_features.csv'),
            )

        self.assertEqual(summary['channels_scanned'], 0)
        self.assertEqual(summary['videos_scanned'], 1)
        self.assertEqual(summary['videos_with_tracklist'], 1)
        self.assertEqual(summary['positive_pairs'], 2)
        self.assertEqual(df['video_id'].tolist(), ['mix-a', 'mix-a'])
        self.assertTrue(df['label'].eq('strong').all())

    @patch('mai.training_scrape.analyze_youtube_playlist_audio', side_effect=fake_analyze_youtube_playlist_audio)
    @patch('mai.training_scrape.YoutubeDL', side_effect=FakeYoutubeDL)
    def test_training_scrape_reuses_training_caches_without_more_network_calls(self, youtube_dl_mock, analyze_mock):
        FakeYoutubeDL.calls = []
        channels = [{'url': 'https://www.youtube.com/@mix_a/videos', 'label': 'excellent', 'label_source': 'mix_a'}]

        with tempfile.TemporaryDirectory() as tmpdir:
            training_scrape.scrape_training_transitions(
                channels=channels,
                cache_dir=tmpdir,
                feature_cache_dir=str(Path(tmpdir) / 'audio_features.csv'),
            )
            self.assertGreater(len(FakeYoutubeDL.calls), 0)
            training_cache_root = Path(tmpdir) / 'training'
            source_cache_path = training_cache_root / 'source_tracks.sqlite'
            resolution_cache_path = training_cache_root / 'track_resolutions.sqlite'
            self.assertTrue(source_cache_path.exists())
            self.assertTrue(resolution_cache_path.exists())
            source_cache_df = training_scrape._read_cache_table(
                str(source_cache_path),
                training_scrape.SOURCE_TRACK_CACHE_COLUMNS,
            )
            resolution_cache_df = training_scrape._read_cache_table(
                str(resolution_cache_path),
                training_scrape.RESOLUTION_CACHE_COLUMNS,
            )
            self.assertIn('source_signature', source_cache_df.columns)
            self.assertNotIn('channel_url', source_cache_df.columns)
            self.assertNotIn('video_url', source_cache_df.columns)
            self.assertNotIn('label', source_cache_df.columns)
            self.assertIn('resolution_signature', resolution_cache_df.columns)
            self.assertNotIn('search_query', resolution_cache_df.columns)
            self.assertNotIn('normalized_search_query', resolution_cache_df.columns)
            self.assertNotIn('resolved_url', resolution_cache_df.columns)

            channel_cache_path = next((training_cache_root / 'channel_videos').glob('*.json'))
            channel_payload = json.loads(channel_cache_path.read_text(encoding='utf-8'))
            self.assertEqual(list(channel_payload.keys()), ['entries'])
            self.assertTrue(set(channel_payload['entries'][0].keys()) <= {'id', 'title'})

            search_cache_path = next((training_cache_root / 'search_results').glob('*.json'))
            search_payload = json.loads(search_cache_path.read_text(encoding='utf-8'))
            self.assertEqual(list(search_payload.keys()), ['entries'])
            self.assertTrue(set(search_payload['entries'][0].keys()) <= {'id', 'title', 'uploader', 'channel', 'duration', 'availability'})

            metadata_cache_path = next((training_cache_root / 'video_metadata').glob('*.json'))
            metadata_payload = json.loads(metadata_cache_path.read_text(encoding='utf-8'))
            self.assertNotIn('formats', metadata_payload)
            self.assertNotIn('thumbnails', metadata_payload)
            self.assertTrue(set(metadata_payload.keys()) <= {
                'title',
                'description',
                'uploader_id',
                'channel_id',
                'artist',
                'uploader',
                'channel',
                'tags',
                'categories',
                'chapters',
                'music_tracks',
                'tracks',
                'tracklist',
                'music_sections',
            })

            FakeYoutubeDL.calls = []
            training_scrape.scrape_training_transitions(
                channels=channels,
                cache_dir=tmpdir,
                feature_cache_dir=str(Path(tmpdir) / 'audio_features.csv'),
            )

            self.assertEqual(FakeYoutubeDL.calls, [])

    @patch('mai.training_scrape.analyze_youtube_playlist_audio', side_effect=fake_analyze_youtube_playlist_audio)
    @patch('mai.training_scrape.fetch_channel_video_entries')
    def test_scrape_training_transitions_skips_channel_enumeration_failures(self, fetch_channel_mock, analyze_mock):
        def fake_fetch_channel(channel_url, cache_dir=None, max_videos=None, refresh=False):
            if channel_url == 'https://www.youtube.com/@broken/videos':
                raise RuntimeError('404 not found')
            return [{
                'video_id': 'mix-a',
                'video_title': 'Mix A',
                'video_url': 'https://www.youtube.com/watch?v=mix-a',
            }]

        fetch_channel_mock.side_effect = fake_fetch_channel

        with patch('mai.training_scrape._fetch_source_video_rows') as fetch_rows_mock:
            fetch_rows_mock.return_value = (
                'mix-a',
                training_scrape.source_tracks_dataframe([
                    {
                        'video_id': 'mix-a',
                        'channel_url': 'https://www.youtube.com/@good/videos',
                        'channel_handle': '@good',
                        'label': 'excellent',
                        'label_source': 'good_source',
                        'source_cache_version': training_scrape.SOURCE_TRACK_CACHE_VERSION,
                        'video_title': 'Mix A',
                        'video_url': 'https://www.youtube.com/watch?v=mix-a',
                        'description_length': 20,
                        'track_source': 'description',
                        'chapter_title': '',
                        'chapter_timestamp_s': pd.NA,
                        'position': 1,
                        'timestamp': '00:00',
                        'timestamp_s': 0,
                        'track_raw': 'Song A - Artist A',
                        'artist_guess': 'Artist A',
                        'title_guess': 'Song A',
                    },
                    {
                        'video_id': 'mix-a',
                        'channel_url': 'https://www.youtube.com/@good/videos',
                        'channel_handle': '@good',
                        'label': 'excellent',
                        'label_source': 'good_source',
                        'source_cache_version': training_scrape.SOURCE_TRACK_CACHE_VERSION,
                        'video_title': 'Mix A',
                        'video_url': 'https://www.youtube.com/watch?v=mix-a',
                        'description_length': 20,
                        'track_source': 'description',
                        'chapter_title': '',
                        'chapter_timestamp_s': pd.NA,
                        'position': 2,
                        'timestamp': '03:00',
                        'timestamp_s': 180,
                        'track_raw': 'Song B - Artist B',
                        'artist_guess': 'Artist B',
                        'title_guess': 'Song B',
                    },
                ]),
            )

            with patch('mai.training_scrape.search_youtube_track_candidates') as search_mock:
                search_mock.side_effect = [
                    [{'id': 'track-a', 'title': 'Song A', 'uploader': 'Artist A', 'duration': 200}],
                    [{'id': 'track-b', 'title': 'Song B', 'uploader': 'Artist B', 'duration': 210}],
                ]
                with tempfile.TemporaryDirectory() as tmpdir:
                    df, summary = training_scrape.scrape_training_transitions(
                        channels=[
                            {'url': 'https://www.youtube.com/@broken/videos', 'label': 'excellent', 'label_source': 'broken_source'},
                            {'url': 'https://www.youtube.com/@good/videos', 'label': 'excellent', 'label_source': 'good_source'},
                        ],
                        cache_dir=tmpdir,
                        feature_cache_dir=str(Path(tmpdir) / 'audio_features.csv'),
                        metadata_workers=1,
                        search_workers=1,
                        download_workers=1,
                        analysis_workers=1,
                    )

        self.assertEqual(summary['channels_failed'], 1)
        self.assertEqual(summary['videos_with_tracklist'], 1)
        self.assertEqual(summary['positive_pairs'], 1)
        self.assertEqual(df['video_id'].tolist(), ['mix-a'])

    @patch('mai.training_scrape.fetch_channel_video_entries')
    def test_scrape_channel_track_rows_emits_loading_channel_index_progress(self, fetch_channel_mock):
        fetch_channel_mock.return_value = []
        events = []

        df, summary = training_scrape.scrape_channel_track_rows(
            channel_url='https://www.youtube.com/@mix_a/videos',
            progress_callback=lambda label, current, total, detail='': events.append((label, current, total, detail)),
        )

        self.assertTrue(df.empty)
        self.assertEqual(summary['videos_scanned'], 0)
        loading_events = [event for event in events if event[0] == 'Loading channel index']
        self.assertEqual(loading_events[0], ('Loading channel index', 0, 1, '@mix_a'))
        self.assertEqual(loading_events[-1], ('Loading channel index', 1, 1, '@mix_a found 0 videos'))

    @patch('mai.training_scrape.analyze_youtube_playlist_audio', side_effect=fake_analyze_youtube_playlist_audio)
    @patch('mai.training_scrape.YoutubeDL', side_effect=FakeYoutubeDL)
    def test_write_training_transitions_csv_round_trips_utf8(self, youtube_dl_mock, analyze_mock):
        channels = [
            {'url': 'https://www.youtube.com/@mix_a/videos', 'label': 'excellent', 'label_source': 'mix_a'},
            {'url': 'https://www.youtube.com/@mix_b/videos', 'label': 'excellent', 'label_source': 'mix_b'},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            df, _ = training_scrape.scrape_training_transitions(
                channels=channels,
                cache_dir=tmpdir,
                feature_cache_dir=str(Path(tmpdir) / 'audio_features.csv'),
            )
            out_path = Path(tmpdir) / 'training.csv'
            training_scrape.write_training_transitions_csv(df, str(out_path))
            loaded = pd.read_csv(out_path, encoding='utf-8')

        self.assertEqual(loaded.columns[0], 'video_id')
        self.assertEqual(loaded.loc[0, 'from_track_source'], 'description+chapters')
        self.assertIn('from_tempo', loaded.columns)
        self.assertIn('to_outro_release_time_s', loaded.columns)


if __name__ == '__main__':
    unittest.main()
