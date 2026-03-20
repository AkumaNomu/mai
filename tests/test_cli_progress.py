import io
import unittest

from mai.cli_progress import CliProgressRenderer


class FakeTtyStringIO(io.StringIO):
    def isatty(self) -> bool:
        return True


class CliProgressRendererTests(unittest.TestCase):
    def test_update_persists_track_notes_for_audio_progress(self):
        stream = FakeTtyStringIO()
        progress = CliProgressRenderer(stream=stream, color=False, heartbeat_interval=0)

        try:
            progress.update('Scanning audio cache', 1, 2, 'video-1 | Artist 1 - Track 1 queued for analysis')
            progress.update('Scanning audio cache', 1, 2, 'video-1 | Artist 1 - Track 1 queued for analysis')
            progress.update('Downloading audio', 1, 2, 'video-1 | Artist 1 - Track 1 downloaded')
            progress.update('Analyzing audio', 1, 2, 'video-1 | Artist 1 - Track 1 analyzed')
        finally:
            progress.close()

        rendered = stream.getvalue()
        self.assertEqual(rendered.count('TRACK video-1 | Artist 1 - Track 1 queued for analysis'), 1)
        self.assertIn('TRACK video-1 | Artist 1 - Track 1 downloaded', rendered)
        self.assertIn('TRACK video-1 | Artist 1 - Track 1 analyzed', rendered)
        self.assertIn('Scanning audio cache: [', rendered)
        self.assertIn('Downloading audio: [', rendered)
        self.assertIn('Analyzing audio: [', rendered)


if __name__ == '__main__':
    unittest.main()
