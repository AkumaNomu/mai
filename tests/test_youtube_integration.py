import tempfile
import unittest
from unittest.mock import patch

from mai import youtube_integration


class FakeYoutubePlaylistDL:
    calls = []

    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def extract_info(self, url, download=False):
        FakeYoutubePlaylistDL.calls.append((url, dict(self.opts)))
        self.last_url = url
        return {
            'title': 'Focus Playlist',
            'entries': [
                {
                    'id': 'track-1',
                    'title': 'Song 1',
                    'uploader': 'Artist 1',
                    'channel': 'Artist 1 Channel',
                    'duration': 180,
                }
            ],
        }


class YoutubeIntegrationTests(unittest.TestCase):
    @patch('mai.youtube_integration.YoutubeDL', side_effect=FakeYoutubePlaylistDL)
    def test_fetch_youtube_playlist_tracks_preserves_source_playlist_title_through_cache(self, youtube_dl_mock):
        FakeYoutubePlaylistDL.calls = []
        with tempfile.TemporaryDirectory() as tmpdir:
            first_df = youtube_integration.fetch_youtube_playlist_tracks('PL123', cache_dir=tmpdir)
            second_df = youtube_integration.fetch_youtube_playlist_tracks('PL123', cache_dir=tmpdir)

        self.assertEqual(first_df.attrs.get('source_playlist_title'), 'Focus Playlist')
        self.assertEqual(second_df.attrs.get('source_playlist_title'), 'Focus Playlist')
        self.assertNotIn('source_playlist_title', first_df.columns)
        self.assertNotIn('source_playlist_title', second_df.columns)
        self.assertEqual(youtube_dl_mock.call_count, 1)

    @patch(
        'mai.youtube_integration.apply_yt_dlp_auth_options',
        side_effect=lambda opts: {**opts, 'cookiefile': 'cookies.txt'},
    )
    @patch('mai.youtube_integration.YoutubeDL', side_effect=FakeYoutubePlaylistDL)
    def test_fetch_youtube_playlist_tracks_passes_cookiefile_to_yt_dlp(self, youtube_dl_mock, auth_mock):
        FakeYoutubePlaylistDL.calls = []
        with tempfile.TemporaryDirectory() as tmpdir:
            youtube_integration.fetch_youtube_playlist_tracks('PL123', cache_dir=tmpdir, refresh=True)

        self.assertEqual(youtube_dl_mock.call_count, 1)
        self.assertEqual(auth_mock.call_count, 1)
        self.assertEqual(FakeYoutubePlaylistDL.calls[0][1].get('cookiefile'), 'cookies.txt')


if __name__ == '__main__':
    unittest.main()
