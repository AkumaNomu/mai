import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

try:
    from google_auth_oauthlib.flow import WSGITimeoutError
    from mai import youtube_export
    GOOGLE_OAUTH_AVAILABLE = True
except ModuleNotFoundError:
    WSGITimeoutError = RuntimeError
    youtube_export = None
    GOOGLE_OAUTH_AVAILABLE = False


class DummyCredentials:
    def __init__(self, payload: str = '{"token": "abc"}'):
        self.payload = payload
        self.valid = True
        self.expired = False
        self.refresh_token = None

    def to_json(self) -> str:
        return self.payload


@unittest.skipUnless(GOOGLE_OAUTH_AVAILABLE, 'google-auth-oauthlib is not installed')
class YoutubeExportAuthTests(unittest.TestCase):
    def test_run_local_server_authorization_uses_ipv4_loopback(self):
        flow = Mock()
        expected_credentials = DummyCredentials()
        flow.run_local_server.return_value = expected_credentials

        credentials = youtube_export._run_local_server_authorization(flow)

        self.assertIs(credentials, expected_credentials)
        kwargs = flow.run_local_server.call_args.kwargs
        self.assertEqual(kwargs['host'], youtube_export.OAUTH_CALLBACK_HOST)
        self.assertEqual(kwargs['port'], 0)
        self.assertEqual(kwargs['timeout_seconds'], youtube_export.OAUTH_LOCAL_SERVER_TIMEOUT_SECONDS)
        self.assertEqual(kwargs['access_type'], 'offline')
        self.assertEqual(kwargs['prompt'], 'consent')

    @patch('mai.youtube_export._run_manual_authorization')
    @patch('mai.youtube_export._run_local_server_authorization')
    @patch('mai.youtube_export._build_youtube_flow')
    def test_authorize_user_credentials_falls_back_to_manual_after_timeout(
        self,
        build_flow_mock,
        local_auth_mock,
        manual_auth_mock,
    ):
        local_flow = Mock(name='local_flow')
        manual_flow = Mock(name='manual_flow')
        build_flow_mock.side_effect = [local_flow, manual_flow]
        local_auth_mock.side_effect = WSGITimeoutError('timeout')
        expected_credentials = DummyCredentials('{"token": "manual"}')
        manual_auth_mock.return_value = expected_credentials

        credentials = youtube_export._authorize_user_credentials(Path('client.json'))

        self.assertIs(credentials, expected_credentials)
        local_auth_mock.assert_called_once_with(local_flow)
        manual_auth_mock.assert_called_once_with(manual_flow)

    @patch('mai.youtube_export.build')
    @patch('mai.youtube_export._authorize_user_credentials')
    def test_load_youtube_service_writes_token_and_builds_client(self, authorize_mock, build_mock):
        expected_credentials = DummyCredentials('{"token": "saved"}')
        authorize_mock.return_value = expected_credentials
        build_mock.return_value = object()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            client_secrets = tmp_path / 'client.json'
            token_file = tmp_path / 'token.json'
            client_secrets.write_text('{"installed": {"client_id": "x"}}', encoding='utf-8')

            youtube_export.load_youtube_service(
                client_secrets_path=str(client_secrets),
                token_path=str(token_file),
            )

            self.assertEqual(token_file.read_text(encoding='utf-8'), '{"token": "saved"}')
            build_mock.assert_called_once_with(
                'youtube',
                'v3',
                credentials=expected_credentials,
                cache_discovery=False,
            )


@unittest.skipUnless(GOOGLE_OAUTH_AVAILABLE, 'google-auth-oauthlib is not installed')
class YoutubeExportUrlTests(unittest.TestCase):
    def test_normalize_authorization_response_upgrades_http_scheme(self):
        response = youtube_export._normalize_authorization_response(
            'http://127.0.0.1:49152/?state=abc&code=xyz',
            'http://127.0.0.1:49152/',
        )
        self.assertTrue(response.startswith('https://127.0.0.1:49152/'))

    def test_normalize_authorization_response_rejects_missing_code(self):
        with self.assertRaisesRegex(ValueError, 'missing the OAuth code'):
            youtube_export._normalize_authorization_response(
                'http://127.0.0.1:49152/?state=abc',
                'http://127.0.0.1:49152/',
            )


if __name__ == '__main__':
    unittest.main()
