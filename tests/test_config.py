import tempfile
import unittest
from pathlib import Path

import run
from mai import training_scrape
from mai.config import get_config_value, load_project_config


class ConfigLoadingTests(unittest.TestCase):
    def test_load_project_config_merges_toml_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'mai.toml'
            config_path.write_text(
                '\n'.join([
                    '[cache]',
                    'root_dir = "custom/cache"',
                    '',
                    '[analysis]',
                    'refresh_cache = true',
                    'download_workers = 3',
                    'analysis_workers = 2',
                    'delete_audio_after_analysis = false',
                    '',
                    '[training]',
                    'output_path = "custom/training.csv"',
                    'metadata_workers = 6',
                    '',
                    '[[training.channels]]',
                    'name = "one"',
                    'url = "https://www.youtube.com/@one/videos"',
                    '',
                    '[[training.channels]]',
                    'name = "two"',
                    'url = "https://www.youtube.com/@two/videos"',
                    '',
                    '[[training.videos]]',
                    'url = "https://www.youtube.com/watch?v=abc123def45"',
                ]),
                encoding='utf-8',
            )

            config = load_project_config(str(config_path))

        self.assertEqual(get_config_value(config, 'cache.root_dir'), 'custom/cache')
        self.assertTrue(get_config_value(config, 'analysis.refresh_cache'))
        self.assertEqual(get_config_value(config, 'analysis.download_workers'), 3)
        self.assertEqual(get_config_value(config, 'analysis.analysis_workers'), 2)
        self.assertFalse(get_config_value(config, 'analysis.delete_audio_after_analysis'))
        self.assertEqual(get_config_value(config, 'training.output_path'), 'custom/training.csv')
        self.assertEqual(get_config_value(config, 'training.metadata_workers'), 6)
        self.assertEqual(get_config_value(config, 'training.max_search_results'), 5)
        self.assertEqual(len(get_config_value(config, 'training.channels')), 2)
        self.assertEqual(len(get_config_value(config, 'training.videos')), 1)

    def test_load_project_config_can_ignore_toml_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'mai.toml'
            config_path.write_text('[analysis]\nrefresh_cache = true\n', encoding='utf-8')

            config = load_project_config(str(config_path), use_config=False)

        self.assertFalse(get_config_value(config, 'analysis.refresh_cache'))


class ConfiguredCliParserTests(unittest.TestCase):
    def test_run_parser_uses_config_defaults_and_cli_overrides(self):
        config = {
            'cache': {'root_dir': 'config/cache', 'audio_dir': 'config/audio'},
            'analysis': {'refresh_cache': True, 'no_audio_analysis': True, 'edge_seconds': 12.5, 'silence_top_db': 28.0, 'flow_profile': 'standard', 'download_workers': 6, 'analysis_workers': 3, 'delete_audio_after_analysis': True},
            'generation': {'allow_reuse': True, 'rate_transitions': True, 'print_recommended_order': True, 'beam_width': 3, 'candidate_width': 7},
            'exports': {
                'ytmusic': {'auth_path': 'config/ytmusic.json', 'title': 'Config Mix', 'privacy': 'PUBLIC'},
                'youtube': {'client_secrets_path': 'config/client.json', 'token_path': 'config/token.json', 'title': 'Config Mix', 'privacy': 'private'},
            },
            'logging': {'level': 'DEBUG'},
        }

        parser = run.build_parser(config, config_path='custom.toml', no_config=False)
        args = parser.parse_args([
            '--csv', 'playlist.csv',
            '--no-refresh-cache',
            '--audio-analysis',
            '--keep-audio-cache',
            '--no-allow-reuse',
            '--no-rate-transitions',
            '--no-print-recommended-order',
        ])

        self.assertEqual(args.config, 'custom.toml')
        self.assertEqual(args.cache_dir, 'config/cache')
        self.assertEqual(args.audio_cache, 'config/audio')
        self.assertEqual(args.edge_seconds, 12.5)
        self.assertEqual(args.flow_profile, 'standard')
        self.assertEqual(args.download_workers, 6)
        self.assertEqual(args.analysis_workers, 3)
        self.assertEqual(args.beam_width, 3)
        self.assertEqual(args.candidate_width, 7)
        self.assertFalse(args.refresh_cache)
        self.assertFalse(args.no_audio_analysis)
        self.assertFalse(args.delete_audio_after_analysis)
        self.assertFalse(args.allow_reuse)
        self.assertFalse(args.rate_transitions)
        self.assertFalse(args.print_recommended_order)

    def test_training_parser_uses_config_defaults_and_cli_overrides(self):
        config = {
            'cache': {'root_dir': 'config/cache', 'audio_dir': 'config/audio'},
            'analysis': {'refresh_cache': True, 'edge_seconds': 16.0, 'silence_top_db': 30.0, 'flow_profile': 'standard', 'download_workers': 5, 'analysis_workers': 2, 'delete_audio_after_analysis': False},
            'training': {'output_path': 'config/training.csv', 'max_videos': 9, 'max_search_results': 11, 'metadata_workers': 6, 'search_workers': 7},
            'logging': {'level': 'DEBUG'},
        }

        parser = training_scrape.build_parser(config, config_path='custom.toml', no_config=False)
        args = parser.parse_args(['--no-refresh-cache', '--delete-audio-after-analysis'])

        self.assertEqual(args.config, 'custom.toml')
        self.assertEqual(args.out, 'config/training.csv')
        self.assertEqual(args.max_videos, 9)
        self.assertEqual(args.max_search_results, 11)
        self.assertEqual(args.metadata_workers, 6)
        self.assertEqual(args.search_workers, 7)
        self.assertEqual(args.cache_dir, 'config/cache')
        self.assertEqual(args.audio_cache, 'config/audio')
        self.assertEqual(args.edge_seconds, 16.0)
        self.assertEqual(args.silence_top_db, 30.0)
        self.assertEqual(args.flow_profile, 'standard')
        self.assertEqual(args.download_workers, 5)
        self.assertEqual(args.analysis_workers, 2)
        self.assertFalse(args.refresh_cache)
        self.assertTrue(args.delete_audio_after_analysis)

    def test_resolve_training_sources_prefers_cli_source_url(self):
        config = {
            'training': {
                'label': 'excellent',
                'label_source': 'configured_source',
                'channels': [{'url': 'https://www.youtube.com/@configured/videos', 'label': 'good', 'label_source': 'configured'}],
            }
        }

        channels = training_scrape.resolve_training_sources(config, channel_url='https://www.youtube.com/watch?v=abc123def45')

        self.assertEqual(len(channels), 1)
        self.assertEqual(channels[0]['url'], 'https://www.youtube.com/watch?v=abc123def45')
        self.assertEqual(channels[0]['label'], 'excellent')
        self.assertEqual(channels[0]['label_source'], 'configured_source')
        self.assertEqual(channels[0]['source_type'], 'video')


if __name__ == '__main__':
    unittest.main()
