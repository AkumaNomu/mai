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


if __name__ == '__main__':
    unittest.main()
