import unittest

import run


class RunTitleTests(unittest.TestCase):
    def test_resolve_youtube_export_base_title_uses_source_playlist_name(self):
        title = run.resolve_youtube_export_base_title('auto', source_playlist_title='Late Night Flow')

        self.assertEqual(title, 'Late Night Flow mai enhanced')

    def test_resolve_youtube_export_base_title_preserves_explicit_title(self):
        title = run.resolve_youtube_export_base_title('Custom Export', source_playlist_title='Late Night Flow')

        self.assertEqual(title, 'Custom Export')


if __name__ == '__main__':
    unittest.main()
