import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from mai import audio_analysis


def sample_features(tempo: float = 120.0) -> dict:
    return {
        'tempo': float(tempo),
        'key': 0,
        'mode': 1,
        'intro_tempo': float(tempo) + 1.0,
        'outro_tempo': float(tempo) - 1.0,
        'intro_seconds_used': 30.0,
        'outro_seconds_used': 30.0,
        'intro_attack_time_s': 1.1,
        'outro_abruptness': 0.42,
        'outro_release_time_s': 0.95,
        'intro_flux_peak': 0.88,
    }


class AudioFeatureCacheTableTests(unittest.TestCase):
    def test_background_resource_profile_caps_workers(self):
        settings = audio_analysis._resolve_analysis_resource_settings(
            download_workers=4,
            analysis_workers=12,
            resource_profile='background',
        )

        self.assertEqual(settings['resource_profile'], 'background')
        self.assertEqual(settings['download_workers'], 1)
        self.assertEqual(settings['analysis_workers'], 1)
        self.assertTrue(settings['force_process_pool'])

    def test_default_resource_profile_preserves_workers(self):
        settings = audio_analysis._resolve_analysis_resource_settings(
            download_workers=3,
            analysis_workers=6,
            resource_profile='default',
        )

        self.assertEqual(settings['resource_profile'], 'default')
        self.assertEqual(settings['download_workers'], 3)
        self.assertEqual(settings['analysis_workers'], 6)
        self.assertFalse(settings['force_process_pool'])

    def test_upsert_and_lookup_feature_cache_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / 'audio_features.csv'
            cache_df, resolved_path = audio_analysis._load_feature_cache_table(str(csv_path))

            self.assertEqual(resolved_path, str(csv_path.with_suffix('.sqlite')))
            cache_df = audio_analysis._upsert_feature_cache_row(
                cache_df,
                str(csv_path),
                'video-1',
                sample_features(),
                edge_seconds=30.0,
                silence_top_db=35.0,
                flow_profile='deep-dj',
            )

            hit = audio_analysis._lookup_feature_cache_row(
                cache_df,
                'video-1',
                edge_seconds=30.0,
                silence_top_db=35.0,
                flow_profile='deep-dj',
            )
            miss = audio_analysis._lookup_feature_cache_row(
                cache_df,
                'video-1',
                edge_seconds=15.0,
                silence_top_db=35.0,
                flow_profile='deep-dj',
            )

            self.assertIsNotNone(hit)
            self.assertEqual(hit['tempo'], 120.0)
            self.assertIsNone(miss)

    def test_upsert_feature_cache_row_keeps_distinct_analysis_signatures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / 'audio_features.csv'
            cache_df = pd.DataFrame()
            cache_df = audio_analysis._upsert_feature_cache_row(
                cache_df,
                str(csv_path),
                'video-1',
                sample_features(tempo=120.0),
                edge_seconds=30.0,
                silence_top_db=35.0,
                flow_profile='deep-dj',
            )
            cache_df = audio_analysis._upsert_feature_cache_row(
                cache_df,
                str(csv_path),
                'video-1',
                sample_features(tempo=130.0),
                edge_seconds=20.0,
                silence_top_db=30.0,
                flow_profile='standard',
            )

            self.assertEqual(len(cache_df), 2)
            self.assertNotIn('edge_seconds', cache_df.columns)
            self.assertNotIn('flow_profile', cache_df.columns)
            standard_row = audio_analysis._lookup_feature_cache_row(
                cache_df,
                'video-1',
                edge_seconds=20.0,
                silence_top_db=30.0,
                flow_profile='standard',
            )
            deep_dj_row = audio_analysis._lookup_feature_cache_row(
                cache_df,
                'video-1',
                edge_seconds=30.0,
                silence_top_db=35.0,
                flow_profile='deep-dj',
            )
            self.assertEqual(float(standard_row['tempo']), 130.0)
            self.assertEqual(float(deep_dj_row['tempo']), 120.0)

    def test_write_feature_cache_table_is_readable_after_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / 'audio_features.csv'
            first_df = pd.DataFrame([
                audio_analysis._feature_cache_record(
                    'video-1',
                    sample_features(tempo=111.0),
                    edge_seconds=30.0,
                    silence_top_db=35.0,
                    flow_profile='deep-dj',
                )
            ])
            second_df = pd.DataFrame([
                audio_analysis._feature_cache_record(
                    'video-2',
                    sample_features(tempo=222.0),
                    edge_seconds=30.0,
                    silence_top_db=35.0,
                    flow_profile='deep-dj',
                )
            ])

            audio_analysis._write_feature_cache_table(first_df, str(csv_path))
            audio_analysis._write_feature_cache_table(second_df, str(csv_path))
            loaded_df = audio_analysis._read_feature_cache_table(str(csv_path))

            self.assertEqual(len(loaded_df), 1)
            self.assertEqual(loaded_df.iloc[0]['video_id'], 'video-2')
            self.assertEqual(float(loaded_df.iloc[0]['tempo']), 222.0)
            self.assertIn('analysis_signature', loaded_df.columns)
            self.assertNotIn('title', loaded_df.columns)
            self.assertNotIn('edge_seconds', loaded_df.columns)
            self.assertEqual(list(Path(tmpdir).glob('audio_features_*.tmp')), [])

    def test_load_feature_cache_table_imports_missing_legacy_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_dir = Path(tmpdir) / 'audio_features'
            legacy_dir.mkdir()
            csv_path = Path(tmpdir) / 'audio_features.csv'

            current_cache_df = pd.DataFrame([
                audio_analysis._feature_cache_record(
                    'video-1',
                    sample_features(tempo=101.0),
                    edge_seconds=30.0,
                    silence_top_db=35.0,
                    flow_profile='deep-dj',
                )
            ])
            audio_analysis._write_feature_cache_table(current_cache_df, str(csv_path))

            legacy_payload = {
                'cache_version': 1,
                'settings': {
                    'edge_seconds': 30.0,
                    'silence_top_db': 35.0,
                    'flow_profile': 'deep-dj',
                },
                'features': sample_features(tempo=202.0),
            }
            (legacy_dir / 'video-1.json').write_text(json.dumps(legacy_payload), encoding='utf-8')
            (legacy_dir / 'video-2.json').write_text(json.dumps(legacy_payload), encoding='utf-8')

            loaded_df, resolved_path = audio_analysis._load_feature_cache_table(str(csv_path))

            self.assertEqual(resolved_path, str(csv_path.with_suffix('.sqlite')))
            self.assertEqual(set(loaded_df['video_id']), {'video-1', 'video-2'})
            preserved = loaded_df.loc[loaded_df['video_id'] == 'video-1'].iloc[0]
            imported = loaded_df.loc[loaded_df['video_id'] == 'video-2'].iloc[0]
            self.assertEqual(float(preserved['tempo']), 101.0)
            self.assertEqual(float(imported['tempo']), 202.0)
            self.assertIn('analysis_signature', loaded_df.columns)
            self.assertNotIn('cache_version', loaded_df.columns)
            self.assertNotIn('description', loaded_df.columns)


class AudioFeatureCacheIntegrationTests(unittest.TestCase):
    def test_download_youtube_audio_passes_cookiefile_to_yt_dlp(self):
        class FakeDownloadYoutubeDL:
            calls = []

            def __init__(self, opts):
                self.opts = opts

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def download(self, urls):
                FakeDownloadYoutubeDL.calls.append(dict(self.opts))
                output_path = str(self.opts['outtmpl']).replace('%(ext)s', 'wav')
                Path(output_path).touch()

        FakeDownloadYoutubeDL.calls = []
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                'mai.audio_analysis.apply_yt_dlp_auth_options',
                side_effect=lambda opts: {**opts, 'cookiefile': 'cookies.txt'},
            ) as auth_mock:
                with patch('mai.audio_analysis.YoutubeDL', side_effect=FakeDownloadYoutubeDL):
                    with patch('mai.audio_analysis._ensure_ffmpeg_dir', return_value=None):
                        audio_path = audio_analysis.download_youtube_audio(
                            'https://www.youtube.com/watch?v=video-1',
                            'video-1',
                            tmpdir,
                        )

        self.assertTrue(audio_path.endswith('video-1.wav'))
        self.assertEqual(auth_mock.call_count, 1)
        self.assertEqual(FakeDownloadYoutubeDL.calls[0].get('cookiefile'), 'cookies.txt')

    @patch('mai.audio_analysis.analyze_audio_file')
    @patch('mai.audio_analysis.download_youtube_audio')
    def test_analyze_youtube_playlist_audio_reuses_then_overwrites_global_csv(
        self,
        download_audio_mock,
        analyze_audio_mock,
    ):
        download_audio_mock.return_value = 'fake.wav'
        analyze_audio_mock.return_value = sample_features(tempo=120.0)
        df = pd.DataFrame([{
            'video_id': 'video-1',
            'title': 'Track 1',
            'artist': 'Artist 1',
            'uploader': 'Artist 1',
            'channel': 'Artist Channel',
            'description': 'Warm and dreamy city pop.',
            'tags': 'city pop, warm',
            'category': 'Music',
            'url': 'https://www.youtube.com/watch?v=video-1',
        }])

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / 'audio_features.csv'
            audio_cache_dir = Path(tmpdir) / 'audio'

            first_result = audio_analysis.analyze_youtube_playlist_audio(
                df,
                audio_cache_dir=str(audio_cache_dir),
                feature_cache_dir=str(csv_path),
                flow_profile='deep-dj',
            )
            self.assertEqual(analyze_audio_mock.call_count, 1)
            self.assertEqual(download_audio_mock.call_count, 1)
            self.assertEqual(float(first_result.loc[0, 'tempo']), 120.0)
            self.assertIn('sentiment_valence', first_result.columns)

            analyze_audio_mock.reset_mock()
            download_audio_mock.reset_mock()
            second_result = audio_analysis.analyze_youtube_playlist_audio(
                df,
                audio_cache_dir=str(audio_cache_dir),
                feature_cache_dir=str(csv_path),
                flow_profile='deep-dj',
            )
            analyze_audio_mock.assert_not_called()
            download_audio_mock.assert_not_called()
            self.assertEqual(float(second_result.loc[0, 'tempo']), 120.0)
            self.assertEqual(second_result.loc[0, 'channel'], 'Artist Channel')

            analyze_audio_mock.return_value = sample_features(tempo=130.0)
            third_result = audio_analysis.analyze_youtube_playlist_audio(
                df,
                audio_cache_dir=str(audio_cache_dir),
                feature_cache_dir=str(csv_path),
                flow_profile='standard',
            )
            self.assertEqual(analyze_audio_mock.call_count, 1)
            self.assertEqual(download_audio_mock.call_count, 1)
            self.assertEqual(float(third_result.loc[0, 'tempo']), 130.0)

            cache_df = audio_analysis._read_feature_cache_table(str(csv_path.with_suffix('.sqlite')))
            self.assertEqual(len(cache_df), 2)
            self.assertEqual(sorted(cache_df['video_id'].tolist()), ['video-1', 'video-1'])
            self.assertIn('analysis_signature', cache_df.columns)
            self.assertNotIn('title', cache_df.columns)
            self.assertNotIn('category', cache_df.columns)
            self.assertIn('sentiment_valence', cache_df.columns)
            deep_dj_cache = audio_analysis._lookup_feature_cache_row(
                cache_df,
                'video-1',
                edge_seconds=30.0,
                silence_top_db=35.0,
                flow_profile='deep-dj',
            )
            standard_cache = audio_analysis._lookup_feature_cache_row(
                cache_df,
                'video-1',
                edge_seconds=30.0,
                silence_top_db=35.0,
                flow_profile='standard',
            )
            self.assertEqual(float(deep_dj_cache['tempo']), 120.0)
            self.assertEqual(float(standard_cache['tempo']), 130.0)

    @patch('mai.audio_analysis.analyze_audio_file')
    @patch('mai.audio_analysis.download_youtube_audio')
    def test_analyze_youtube_playlist_audio_emits_stage_start_progress(
        self,
        download_audio_mock,
        analyze_audio_mock,
    ):
        download_audio_mock.return_value = 'fake.wav'
        analyze_audio_mock.return_value = sample_features(tempo=120.0)
        df = pd.DataFrame([{
            'video_id': 'video-1',
            'title': 'Track 1',
            'artist': 'Artist 1',
            'uploader': 'Artist 1',
            'channel': 'Artist Channel',
            'description': 'Warm and dreamy city pop.',
            'tags': 'city pop, warm',
            'category': 'Music',
            'url': 'https://www.youtube.com/watch?v=video-1',
        }])
        events = []

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_analysis.analyze_youtube_playlist_audio(
                df,
                audio_cache_dir=str(Path(tmpdir) / 'audio'),
                feature_cache_dir=str(Path(tmpdir) / 'audio_features.csv'),
                flow_profile='deep-dj',
                download_workers=1,
                analysis_workers=1,
                progress_callback=lambda label, current, total, detail='': events.append((label, current, total, detail)),
            )

        self.assertIn(('Scanning audio cache', 0, 1, 'checking 1 tracks'), events)
        self.assertIn(('Downloading audio', 0, 1, 'starting 1 downloads with 1 workers'), events)
        self.assertIn(('Analyzing audio', 0, 1, 'starting 1 tracks with 1 workers'), events)
        self.assertIn(
            ('Scanning audio cache', 1, 1, 'video-1 | Artist 1 - Track 1 queued for analysis'),
            events,
        )
        self.assertIn(
            ('Downloading audio', 1, 1, 'video-1 | Artist 1 - Track 1 downloaded'),
            events,
        )
        self.assertIn(
            ('Analyzing audio', 1, 1, 'video-1 | Artist 1 - Track 1 analyzed'),
            events,
        )

    @patch('mai.audio_analysis.analyze_audio_file')
    @patch('mai.audio_analysis.download_youtube_audio')
    def test_analyze_youtube_playlist_audio_dedupes_duplicate_pending_video_ids(
        self,
        download_audio_mock,
        analyze_audio_mock,
    ):
        download_audio_mock.return_value = 'fake.wav'
        analyze_audio_mock.return_value = sample_features(tempo=118.0)
        df = pd.DataFrame([
            {
                'video_id': 'video-dup',
                'title': 'Track 1',
                'artist': 'Artist 1',
                'uploader': 'Artist 1',
                'channel': 'Artist Channel',
                'description': 'first row',
                'tags': 'one',
                'category': 'Music',
                'url': 'https://www.youtube.com/watch?v=video-dup',
            },
            {
                'video_id': 'video-dup',
                'title': 'Track 1',
                'artist': 'Artist 1',
                'uploader': 'Artist 1',
                'channel': 'Artist Channel',
                'description': 'second row',
                'tags': 'two',
                'category': 'Music',
                'url': 'https://www.youtube.com/watch?v=video-dup',
            },
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            result = audio_analysis.analyze_youtube_playlist_audio(
                df,
                audio_cache_dir=str(Path(tmpdir) / 'audio'),
                feature_cache_dir=str(Path(tmpdir) / 'audio_features.csv'),
                download_workers=1,
                analysis_workers=1,
            )

        self.assertEqual(download_audio_mock.call_count, 1)
        self.assertEqual(analyze_audio_mock.call_count, 1)
        self.assertTrue(result['tempo'].notna().all())
        self.assertEqual(result['description'].tolist(), ['first row', 'second row'])

    @patch('mai.audio_analysis.analyze_audio_file')
    @patch('mai.audio_analysis.download_youtube_audio')
    def test_analyze_youtube_playlist_audio_deletes_cached_audio_after_persisting_features(
        self,
        download_audio_mock,
        analyze_audio_mock,
    ):
        analyze_audio_mock.return_value = sample_features(tempo=121.0)
        df = pd.DataFrame([{
            'video_id': 'video-1',
            'title': 'Track 1',
            'artist': 'Artist 1',
            'url': 'https://www.youtube.com/watch?v=video-1',
        }])

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / 'audio' / 'video-1.wav'
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            audio_path.write_bytes(b'fake-audio')
            download_audio_mock.return_value = str(audio_path)

            result = audio_analysis.analyze_youtube_playlist_audio(
                df,
                audio_cache_dir=str(audio_path.parent),
                feature_cache_dir=str(Path(tmpdir) / 'audio_features.csv'),
                delete_audio_after_analysis=True,
                download_workers=1,
                analysis_workers=1,
            )

            self.assertEqual(float(result.loc[0, 'tempo']), 121.0)
            self.assertFalse(audio_path.exists())

    @patch('mai.audio_analysis.analyze_audio_file')
    @patch('mai.audio_analysis.download_youtube_audio')
    def test_analyze_youtube_playlist_audio_migrates_legacy_json_cache(
        self,
        download_audio_mock,
        analyze_audio_mock,
    ):
        df = pd.DataFrame([{'video_id': 'video-1', 'title': 'Track 1'}])

        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_dir = Path(tmpdir) / 'audio_features'
            legacy_dir.mkdir()
            legacy_payload = {
                'cache_version': 1,
                'settings': {
                    'edge_seconds': 30.0,
                    'silence_top_db': 35.0,
                    'flow_profile': 'deep-dj',
                },
                'features': sample_features(tempo=140.0),
            }
            (legacy_dir / 'video-1.json').write_text(json.dumps(legacy_payload), encoding='utf-8')

            result = audio_analysis.analyze_youtube_playlist_audio(
                df,
                audio_cache_dir=str(Path(tmpdir) / 'audio'),
                feature_cache_dir=str(legacy_dir),
                flow_profile='deep-dj',
            )

            analyze_audio_mock.assert_not_called()
            download_audio_mock.assert_not_called()
            self.assertEqual(float(result.loc[0, 'tempo']), 140.0)

            sqlite_path = Path(f'{legacy_dir}.sqlite')
            self.assertTrue(sqlite_path.exists())
            cache_df = audio_analysis._read_feature_cache_table(str(sqlite_path))
            self.assertEqual(len(cache_df), 1)
            self.assertEqual(cache_df.iloc[0]['video_id'], 'video-1')
            self.assertEqual(float(cache_df.iloc[0]['tempo']), 140.0)

    @patch('mai.audio_analysis.analyze_audio_file')
    def test_analyze_audio_cache_directory_persists_and_deletes_audio(
        self,
        analyze_audio_mock,
    ):
        analyze_audio_mock.return_value = sample_features(tempo=123.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_dir = Path(tmpdir) / 'audio'
            audio_dir.mkdir()
            audio_path = audio_dir / 'video-1.wav'
            audio_path.write_bytes(b'cached-audio')
            csv_path = Path(tmpdir) / 'audio_features.csv'

            result_df, summary = audio_analysis.analyze_audio_cache_directory(
                audio_cache_dir=str(audio_dir),
                feature_cache_dir=str(csv_path),
                analysis_workers=1,
                delete_audio_after_analysis=True,
            )

            self.assertEqual(summary['audio_files_found'], 1)
            self.assertEqual(summary['unique_tracks_found'], 1)
            self.assertEqual(summary['analyzed'], 1)
            self.assertEqual(summary['audio_files_deleted'], 1)
            self.assertFalse(audio_path.exists())
            self.assertEqual(analyze_audio_mock.call_count, 1)
            self.assertEqual(result_df.loc[0, 'analysis_status'], 'analyzed')
            self.assertEqual(float(result_df.loc[0, 'tempo']), 123.0)

            cache_df = audio_analysis._read_feature_cache_table(str(csv_path.with_suffix('.sqlite')))
            self.assertEqual(len(cache_df), 1)
            self.assertEqual(cache_df.iloc[0]['video_id'], 'video-1')
            self.assertEqual(float(cache_df.iloc[0]['tempo']), 123.0)

    @patch('mai.audio_analysis.analyze_audio_file')
    def test_analyze_audio_cache_directory_reuses_cache_and_deletes_redundant_audio(
        self,
        analyze_audio_mock,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_dir = Path(tmpdir) / 'audio'
            audio_dir.mkdir()
            audio_path = audio_dir / 'video-1.wav'
            audio_path.write_bytes(b'cached-audio')
            csv_path = Path(tmpdir) / 'audio_features.csv'

            cache_df = pd.DataFrame([
                audio_analysis._feature_cache_record(
                    'video-1',
                    sample_features(tempo=141.0),
                    edge_seconds=30.0,
                    silence_top_db=35.0,
                    flow_profile='deep-dj',
                )
            ])
            audio_analysis._write_feature_cache_table(cache_df, str(csv_path))

            result_df, summary = audio_analysis.analyze_audio_cache_directory(
                audio_cache_dir=str(audio_dir),
                feature_cache_dir=str(csv_path),
                analysis_workers=1,
                delete_audio_after_analysis=True,
            )

            analyze_audio_mock.assert_not_called()
            self.assertEqual(summary['cache_hits'], 1)
            self.assertEqual(summary['analyzed'], 0)
            self.assertEqual(summary['audio_files_deleted'], 1)
            self.assertFalse(audio_path.exists())
            self.assertEqual(result_df.loc[0, 'analysis_status'], 'cached')
            self.assertEqual(float(result_df.loc[0, 'tempo']), 141.0)


if __name__ == '__main__':
    unittest.main()
