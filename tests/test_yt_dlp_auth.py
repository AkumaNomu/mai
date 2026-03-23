import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mai import yt_dlp_auth


VALID_COOKIE_ROW = '.youtube.com\tTRUE\t/\tTRUE\t1789437040\tVISITOR_INFO1_LIVE\txblzUoo2WmI\n'


class YtDlpAuthTests(unittest.TestCase):
    @patch('mai.yt_dlp_auth.ensure_yt_dlp_js_runtime', return_value={})
    @patch('mai.yt_dlp_auth.ensure_yt_dlp_ffmpeg_location', return_value=None)
    def test_resolve_yt_dlp_cookiefile_uses_env_override(self, ffmpeg_mock, js_runtime_mock):
        with tempfile.TemporaryDirectory() as tmpdir:
            cookie_path = os.path.join(tmpdir, 'cookies.txt')
            with open(cookie_path, 'w', encoding='utf-8') as handle:
                handle.write('# Netscape HTTP Cookie File\n')
                handle.write(VALID_COOKIE_ROW)
            with patch.dict(os.environ, {'MAI_YTDLP_COOKIEFILE': cookie_path}, clear=False):
                resolved = yt_dlp_auth.resolve_yt_dlp_cookiefile()

        self.assertEqual(resolved, cookie_path)

    @patch('mai.yt_dlp_auth.ensure_yt_dlp_js_runtime', return_value={})
    @patch('mai.yt_dlp_auth.ensure_yt_dlp_ffmpeg_location', return_value=None)
    def test_apply_yt_dlp_auth_options_adds_cookiefile_when_file_exists(self, ffmpeg_mock, js_runtime_mock):
        with tempfile.TemporaryDirectory() as tmpdir:
            cookie_path = os.path.join(tmpdir, 'cookies.txt')
            with open(cookie_path, 'w', encoding='utf-8') as handle:
                handle.write('# Netscape HTTP Cookie File\n')
                handle.write(VALID_COOKIE_ROW)

            opts = yt_dlp_auth.apply_yt_dlp_auth_options({'quiet': True}, cookiefile=cookie_path)

        self.assertEqual(opts['cookiefile'], cookie_path)
        self.assertTrue(opts['quiet'])

    @patch('mai.yt_dlp_auth.ensure_yt_dlp_js_runtime', return_value={})
    @patch('mai.yt_dlp_auth.ensure_yt_dlp_ffmpeg_location', return_value=None)
    def test_apply_yt_dlp_auth_options_skips_missing_cookiefile(self, ffmpeg_mock, js_runtime_mock):
        opts = yt_dlp_auth.apply_yt_dlp_auth_options({'quiet': True}, cookiefile='missing-cookies.txt')

        self.assertNotIn('cookiefile', opts)
        self.assertIs(opts['cachedir'], False)
        self.assertTrue(opts['quiet'])

    @patch('mai.yt_dlp_auth.ensure_yt_dlp_js_runtime', return_value={'node': {'path': 'C:\\node.exe'}})
    @patch('mai.yt_dlp_auth.ensure_yt_dlp_ffmpeg_location', return_value=None)
    def test_apply_yt_dlp_auth_options_skips_js_runtime_setup_when_player_skip_js(self, ffmpeg_mock, js_runtime_mock):
        opts = yt_dlp_auth.apply_yt_dlp_auth_options({
            'quiet': True,
            'extractor_args': {'youtube': {'player_skip': ['js']}},
        })

        self.assertNotIn('js_runtimes', opts)
        self.assertNotIn('remote_components', opts)
        self.assertEqual(opts['extractor_args'], {'youtube': {'player_skip': ['js'], 'player_client': ['android_vr']}})
        js_runtime_mock.assert_not_called()

    @patch('mai.yt_dlp_auth.ensure_yt_dlp_js_runtime', return_value={})
    @patch('mai.yt_dlp_auth.ensure_yt_dlp_ffmpeg_location', return_value=None)
    def test_apply_yt_dlp_auth_options_allows_internal_cache_with_env_override(self, ffmpeg_mock, js_runtime_mock):
        with patch.dict(os.environ, {'MAI_YTDLP_ENABLE_INTERNAL_CACHE': '1'}, clear=False):
            opts = yt_dlp_auth.apply_yt_dlp_auth_options({'quiet': True})

        self.assertNotEqual(opts['cachedir'], False)
        self.assertIn('yt_dlp_internal', opts['cachedir'])

    @patch('mai.yt_dlp_auth.ensure_yt_dlp_js_runtime', return_value={})
    @patch('mai.yt_dlp_auth.ensure_yt_dlp_ffmpeg_location', return_value=None)
    def test_resolve_yt_dlp_cookiefile_repairs_split_cookie_rows(self, ffmpeg_mock, js_runtime_mock):
        with tempfile.TemporaryDirectory() as tmpdir:
            cookie_path = Path(tmpdir) / 'cookies.txt'
            cookie_path.write_text(
                '# Netscape HTTP Cookie File\n'
                'accounts.google.com\tFALSE\t/\tTRUE\n'
                '\t1808089574\t__Host-1PLSID\tcookie-value\n'
                '.youtube.com\tTRUE\t/\tTRUE\t1789437040\tVISITOR_INFO1_LIVE\txblzUoo2WmI\n',
                encoding='utf-8',
            )

            resolved = yt_dlp_auth.resolve_yt_dlp_cookiefile(str(cookie_path))
            resolved_lines = Path(resolved).read_text(encoding='utf-8').splitlines()

        self.assertNotEqual(resolved, str(cookie_path))
        data_lines = [line for line in resolved_lines if line and not line.startswith('#')]
        self.assertEqual(len(data_lines), 2)
        self.assertEqual(len(data_lines[0].split('\t')), 7)
        self.assertIn('__Host-1PLSID', data_lines[0])

    @patch('mai.yt_dlp_auth.ensure_yt_dlp_js_runtime', return_value={})
    @patch('mai.yt_dlp_auth.ensure_yt_dlp_ffmpeg_location', return_value=None)
    def test_resolve_yt_dlp_cookiefile_drops_unrepairable_invalid_lines(self, ffmpeg_mock, js_runtime_mock):
        with tempfile.TemporaryDirectory() as tmpdir:
            cookie_path = Path(tmpdir) / 'cookies.txt'
            cookie_path.write_text(
                '# Netscape HTTP Cookie File\n'
                'broken\tcookie\trow\ttoo-short\n'
                '.youtube.com\tTRUE\t/\tTRUE\t1789437040\tVISITOR_INFO1_LIVE\txblzUoo2WmI\n',
                encoding='utf-8',
            )

            resolved = yt_dlp_auth.resolve_yt_dlp_cookiefile(str(cookie_path))
            resolved_lines = Path(resolved).read_text(encoding='utf-8').splitlines()

        self.assertNotEqual(resolved, str(cookie_path))
        data_lines = [line for line in resolved_lines if line and not line.startswith('#')]
        self.assertEqual(len(data_lines), 1)
        self.assertEqual(len(data_lines[0].split('\t')), 7)

    @patch('mai.yt_dlp_auth.ensure_yt_dlp_js_runtime', return_value={})
    @patch('mai.yt_dlp_auth.ensure_yt_dlp_ffmpeg_location', return_value=None)
    def test_resolve_yt_dlp_cookiefile_returns_none_for_empty_cookie_file(self, ffmpeg_mock, js_runtime_mock):
        with tempfile.TemporaryDirectory() as tmpdir:
            cookie_path = Path(tmpdir) / 'cookies.txt'
            cookie_path.write_text('', encoding='utf-8')

            resolved = yt_dlp_auth.resolve_yt_dlp_cookiefile(str(cookie_path))

        self.assertIsNone(resolved)

    @patch('mai.yt_dlp_auth.ensure_yt_dlp_js_runtime', return_value={})
    @patch('mai.yt_dlp_auth.ensure_yt_dlp_ffmpeg_location', return_value=None)
    def test_resolve_yt_dlp_cookiefile_adds_missing_netscape_header(self, ffmpeg_mock, js_runtime_mock):
        with tempfile.TemporaryDirectory() as tmpdir:
            cookie_path = Path(tmpdir) / 'cookies.txt'
            cookie_path.write_text(VALID_COOKIE_ROW, encoding='utf-8')

            resolved = yt_dlp_auth.resolve_yt_dlp_cookiefile(str(cookie_path))
            resolved_lines = Path(resolved).read_text(encoding='utf-8').splitlines()

        self.assertNotEqual(resolved, str(cookie_path))
        self.assertEqual(resolved_lines[0], '# Netscape HTTP Cookie File')
        data_lines = [line for line in resolved_lines if line and not line.startswith('#')]
        self.assertEqual(data_lines, [VALID_COOKIE_ROW.strip()])

    @patch('mai.yt_dlp_auth.ensure_yt_dlp_ffmpeg_location', return_value=None)
    @patch('mai.yt_dlp_auth._ensure_windows_deno_runtime', return_value={})
    @patch('mai.yt_dlp_auth.ensure_yt_dlp_js_runtime', return_value={'node': {'path': 'C:\\node.exe'}})
    def test_apply_yt_dlp_auth_options_skips_node_runtime_by_default(self, js_runtime_mock, deno_fallback_mock, ffmpeg_mock):
        opts = yt_dlp_auth.apply_yt_dlp_auth_options({'quiet': True})

        self.assertNotIn('js_runtimes', opts)
        self.assertIn('ejs:github', opts['remote_components'])
        self.assertTrue(opts['quiet'])
        deno_fallback_mock.assert_called_once()

    @patch('mai.yt_dlp_auth.ensure_yt_dlp_js_runtime', return_value={})
    @patch('mai.yt_dlp_auth.ensure_yt_dlp_ffmpeg_location', return_value=None)
    def test_apply_yt_dlp_auth_options_keeps_remote_ejs_fallback_without_runtime(self, ffmpeg_mock, js_runtime_mock):
        opts = yt_dlp_auth.apply_yt_dlp_auth_options({'quiet': True})

        self.assertNotIn('js_runtimes', opts)
        self.assertIn('ejs:github', opts['remote_components'])
        self.assertTrue(opts['quiet'])

    @patch('mai.yt_dlp_auth.ensure_yt_dlp_ffmpeg_location', return_value=None)
    @patch('mai.yt_dlp_auth._ensure_windows_deno_runtime', return_value={'deno': {'path': 'C:\\deno.exe'}})
    @patch('mai.yt_dlp_auth.ensure_yt_dlp_js_runtime', return_value={'node': {'path': 'C:\\node.exe'}})
    def test_apply_yt_dlp_auth_options_allows_node_runtime_with_env_override(self, js_runtime_mock, deno_fallback_mock, ffmpeg_mock):
        with patch.dict(os.environ, {'MAI_YTDLP_ALLOW_NODE_RUNTIME': '1'}, clear=False):
            opts = yt_dlp_auth.apply_yt_dlp_auth_options({'quiet': True})

        self.assertEqual(opts['js_runtimes'], {'node': {'path': 'C:\\node.exe'}})
        self.assertIn('ejs:github', opts['remote_components'])
        self.assertTrue(opts['quiet'])
        deno_fallback_mock.assert_not_called()

    @patch('mai.yt_dlp_auth.ensure_yt_dlp_ffmpeg_location', return_value=None)
    @patch('mai.yt_dlp_auth._ensure_windows_deno_runtime', return_value={'deno': {'path': 'C:\\deno.exe'}})
    @patch('mai.yt_dlp_auth.ensure_yt_dlp_js_runtime', return_value={'node': {'path': 'C:\\node.exe'}})
    def test_apply_yt_dlp_auth_options_uses_deno_fallback_when_only_node_detected(self, js_runtime_mock, deno_fallback_mock, ffmpeg_mock):
        opts = yt_dlp_auth.apply_yt_dlp_auth_options({'quiet': True})

        self.assertEqual(opts['js_runtimes'], {'deno': {'path': 'C:\\deno.exe'}})
        self.assertIn('ejs:github', opts['remote_components'])
        self.assertTrue(opts['quiet'])
        deno_fallback_mock.assert_called_once()

    @patch('mai.yt_dlp_auth.ensure_yt_dlp_js_runtime', return_value={})
    @patch('mai.yt_dlp_auth.ensure_yt_dlp_ffmpeg_location', return_value='data\\tools\\ffmpeg\\bin')
    def test_apply_yt_dlp_auth_options_adds_ffmpeg_location(self, ffmpeg_mock, js_runtime_mock):
        opts = yt_dlp_auth.apply_yt_dlp_auth_options({'quiet': True})

        self.assertEqual(opts['ffmpeg_location'], 'data\\tools\\ffmpeg\\bin')
        self.assertTrue(opts['quiet'])

    @patch('mai.yt_dlp_auth.urllib.request.urlretrieve')
    @patch('mai.yt_dlp_auth.shutil.which', return_value=None)
    @patch('mai.yt_dlp_auth._is_usable_js_runtime_binary', side_effect=lambda path, **kwargs: Path(path).exists())
    def test_ensure_yt_dlp_js_runtime_downloads_local_deno_on_windows_when_missing(self, runtime_ok_mock, which_mock, urlretrieve_mock):
        with tempfile.TemporaryDirectory() as tmpdir:
            def fake_urlretrieve(url, destination, reporthook=None):
                zip_path = Path(destination)
                zip_path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory() as zipdir:
                    deno_path = Path(zipdir) / 'deno.exe'
                    deno_path.write_text('stub', encoding='utf-8')
                    import zipfile
                    with zipfile.ZipFile(zip_path, 'w') as zf:
                        zf.write(deno_path, arcname='deno.exe')
                if callable(reporthook):
                    reporthook(1, 1024, 1024)
                return str(zip_path), None

            urlretrieve_mock.side_effect = fake_urlretrieve
            with patch('mai.yt_dlp_auth.os.name', 'nt'):
                runtimes = yt_dlp_auth.ensure_yt_dlp_js_runtime(tmpdir)

        self.assertEqual(set(runtimes), {'deno'})
        self.assertTrue(runtimes['deno']['path'].endswith('deno.exe'))


if __name__ == '__main__':
    unittest.main()
