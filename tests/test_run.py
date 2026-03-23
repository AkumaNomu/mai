import unittest

import run


class RunTitleTests(unittest.TestCase):
    def test_resolve_youtube_export_base_title_uses_source_playlist_name(self):
        title = run.resolve_youtube_export_base_title('auto', source_playlist_title='Late Night Flow')

        self.assertEqual(title, 'Late Night Flow mai enhanced')

    def test_resolve_youtube_export_base_title_preserves_explicit_title(self):
        title = run.resolve_youtube_export_base_title('Custom Export', source_playlist_title='Late Night Flow')

        self.assertEqual(title, 'Custom Export')

    def test_build_parser_accepts_background_resource_profile(self):
        parser = run.build_parser({}, config_path='mai.toml', no_config=True)

        args = parser.parse_args(['--csv', 'playlist.csv', '--resource-profile', 'background'])

        self.assertEqual(args.resource_profile, 'background')

    def test_build_parser_accepts_transition_model_options(self):
        parser = run.build_parser({}, config_path='mai.toml', no_config=True)

        args = parser.parse_args([
            '--csv', 'playlist.csv',
            '--train-transition-model',
            '--transition-model-path', 'data/cache/model.joblib',
            '--transition-model-out', 'data/cache/out.joblib',
            '--transition-model-weight', '0.25',
            '--transition-model-negative-ratio', '1.5',
            '--transition-model-random-state', '99',
            '--transition-model-device', 'cpu',
        ])

        self.assertTrue(args.train_transition_model)
        self.assertEqual(args.transition_model_path, 'data/cache/model.joblib')
        self.assertEqual(args.transition_model_out, 'data/cache/out.joblib')
        self.assertAlmostEqual(args.transition_model_weight, 0.25)
        self.assertAlmostEqual(args.transition_model_negative_ratio, 1.5)
        self.assertEqual(args.transition_model_random_state, 99)
        self.assertEqual(args.transition_model_device, 'cpu')


if __name__ == '__main__':
    unittest.main()
