import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mai import audio_analysis, cache_cleanup


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


class CacheCleanupTests(unittest.TestCase):
    def test_clean_useless_cache_removes_redundant_audio_and_temp_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / 'cache'
            audio_root = Path(tmpdir) / 'audio'
            feature_csv = cache_root / 'audio_features.csv'
            cache_root.mkdir(parents=True, exist_ok=True)
            audio_root.mkdir(parents=True, exist_ok=True)

            feature_df = pd.DataFrame([
                audio_analysis._feature_cache_record(
                    'video-1',
                    sample_features(),
                    edge_seconds=30.0,
                    silence_top_db=35.0,
                    flow_profile='deep-dj',
                )
            ])
            audio_analysis._write_feature_cache_table(feature_df, str(feature_csv))

            redundant_audio = audio_root / 'video-1.wav'
            redundant_audio.write_bytes(b'12345')
            kept_audio = audio_root / 'video-2.wav'
            kept_audio.write_bytes(b'123')
            temp_audio = audio_root / 'video-3.webm.part'
            temp_audio.write_bytes(b'1234')

            legacy_dir = feature_csv.with_suffix('')
            legacy_dir.mkdir(parents=True, exist_ok=True)
            redundant_json = legacy_dir / 'video-1.json'
            redundant_json.write_text('{"ok": true}', encoding='utf-8')
            kept_json = legacy_dir / 'video-3.json'
            kept_json.write_text('{"ok": true}', encoding='utf-8')

            yt_dlp_dir = cache_root / 'yt_dlp'
            yt_dlp_dir.mkdir(parents=True, exist_ok=True)
            sanitized_cookie = yt_dlp_dir / 'cookies_test_sanitized.txt'
            sanitized_cookie.write_text('cookie', encoding='utf-8')
            keep_file = yt_dlp_dir / 'keep.txt'
            keep_file.write_text('keep', encoding='utf-8')

            summary = cache_cleanup.clean_useless_cache(
                cache_dir=str(cache_root),
                audio_cache_dir=str(audio_root),
                feature_cache_dir=str(feature_csv),
            )
            self.assertEqual(summary['audio_files_deleted'], 1)
            self.assertEqual(summary['audio_temp_files_deleted'], 1)
            self.assertEqual(summary['legacy_feature_json_deleted'], 1)
            self.assertEqual(summary['yt_dlp_cache_files_deleted'], 1)
            self.assertEqual(summary['audio_files_kept'], 1)
            self.assertGreater(summary['bytes_freed'], 0)
            self.assertFalse(redundant_audio.exists())
            self.assertFalse(temp_audio.exists())
            self.assertTrue(kept_audio.exists())
            self.assertFalse(redundant_json.exists())
            self.assertTrue(kept_json.exists())
            self.assertFalse(sanitized_cookie.exists())
            self.assertTrue(keep_file.exists())

    def test_clean_useless_cache_supports_dry_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / 'cache'
            audio_root = Path(tmpdir) / 'audio'
            feature_csv = cache_root / 'audio_features.csv'
            cache_root.mkdir(parents=True, exist_ok=True)
            audio_root.mkdir(parents=True, exist_ok=True)

            feature_df = pd.DataFrame([
                audio_analysis._feature_cache_record(
                    'video-1',
                    sample_features(),
                    edge_seconds=30.0,
                    silence_top_db=35.0,
                    flow_profile='deep-dj',
                )
            ])
            audio_analysis._write_feature_cache_table(feature_df, str(feature_csv))

            redundant_audio = audio_root / 'video-1.wav'
            redundant_audio.write_bytes(b'12345')

            summary = cache_cleanup.clean_useless_cache(
                cache_dir=str(cache_root),
                audio_cache_dir=str(audio_root),
                feature_cache_dir=str(feature_csv),
                dry_run=True,
            )

            self.assertEqual(summary['audio_files_deleted'], 1)
            self.assertTrue(redundant_audio.exists())


if __name__ == '__main__':
    unittest.main()
