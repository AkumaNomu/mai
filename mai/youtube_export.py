from __future__ import annotations

import logging
import socket
import time
import webbrowser
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow, WSGITimeoutError
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


logger = logging.getLogger(__name__)

SCOPES = ['https://www.googleapis.com/auth/youtube']
OAUTH_CALLBACK_HOST = '127.0.0.1'
OAUTH_LOCAL_SERVER_TIMEOUT_SECONDS = 120


def _build_youtube_flow(client_secrets: Path) -> InstalledAppFlow:
    return InstalledAppFlow.from_client_secrets_file(str(client_secrets), SCOPES)


def _run_local_server_authorization(flow: InstalledAppFlow) -> Credentials:
    logger.info(
        'Starting Google OAuth loopback callback on http://%s:<random-port>/',
        OAUTH_CALLBACK_HOST,
    )
    return flow.run_local_server(
        host=OAUTH_CALLBACK_HOST,
        port=0,
        authorization_prompt_message='Open this URL to authorize Mai: {url}',
        success_message='Mai received the Google authorization response. You may close this tab.',
        open_browser=True,
        timeout_seconds=OAUTH_LOCAL_SERVER_TIMEOUT_SECONDS,
        access_type='offline',
        prompt='consent',
    )


def _reserve_loopback_port(host: str = OAUTH_CALLBACK_HOST) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _normalize_authorization_response(authorization_response: str, redirect_uri: str) -> str:
    response = authorization_response.strip()
    if not response:
        raise ValueError('redirect URL is empty')
    if response.startswith('?'):
        response = f'{redirect_uri}{response}'
    elif response.startswith('/'):
        response = f'{redirect_uri.rstrip("/")}{response}'

    parsed = urlparse(response)
    if parsed.scheme not in {'http', 'https'}:
        raise ValueError('paste the full http://127.0.0.1:PORT/... URL from the browser address bar')

    query = parse_qs(parsed.query)
    if 'code' not in query:
        raise ValueError('redirect URL is missing the OAuth code parameter')

    if parsed.scheme == 'http':
        response = parsed._replace(scheme='https').geturl()
    return response


def _run_manual_authorization(flow: InstalledAppFlow) -> Credentials:
    port = _reserve_loopback_port()
    flow.redirect_uri = f'http://{OAUTH_CALLBACK_HOST}:{port}/'
    auth_url, _ = flow.authorization_url(access_type='offline', prompt='consent')

    print(f'Open this URL to authorize Mai: {auth_url}')
    print(
        'If your browser ends on a connection error page after approval, copy the full '
        f'"{flow.redirect_uri}?code=..." URL from the address bar and paste it below.'
    )

    try:
        webbrowser.open(auth_url, new=1, autoraise=True)
    except webbrowser.Error as exc:
        logger.debug('Unable to auto-open browser for manual Google OAuth fallback: %s', exc)

    while True:
        authorization_response = input('Paste the full redirect URL: ').strip()
        try:
            flow.fetch_token(
                authorization_response=_normalize_authorization_response(
                    authorization_response,
                    flow.redirect_uri,
                )
            )
            return flow.credentials
        except ValueError as exc:
            print(f'Invalid redirect URL: {exc}')


def _authorize_user_credentials(client_secrets: Path) -> Credentials:
    try:
        return _run_local_server_authorization(_build_youtube_flow(client_secrets))
    except WSGITimeoutError:
        logger.warning(
            'Timed out waiting for the Google OAuth browser callback. Falling back to manual redirect URL entry.'
        )
        return _run_manual_authorization(_build_youtube_flow(client_secrets))


def load_youtube_service(
    client_secrets_path: str = 'data/youtube_client_secret.json',
    token_path: str = 'data/youtube_token.json',
):
    """Load an authenticated YouTube Data API client using desktop OAuth."""
    client_secrets = Path(client_secrets_path)
    token_file = Path(token_path)
    if not client_secrets.exists():
        raise FileNotFoundError(f'YouTube client secrets file not found: {client_secrets}')

    credentials = None
    if token_file.exists():
        credentials = Credentials.from_authorized_user_file(str(token_file), SCOPES)

    if credentials and credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
    elif not credentials or not credentials.valid:
        credentials = _authorize_user_credentials(client_secrets)
    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.write_text(credentials.to_json(), encoding='utf-8')

    return build('youtube', 'v3', credentials=credentials, cache_discovery=False)


def _execute_with_retry(request_factory, label: str, retries: int = 3, base_delay_s: float = 1.0) -> Any:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return request_factory().execute()
        except HttpError as exc:
            last_error = exc
            status = getattr(getattr(exc, 'resp', None), 'status', None)
            retryable = status in {403, 429, 500, 502, 503, 504}
            if not retryable or attempt >= retries:
                raise
            delay = base_delay_s * (2 ** (attempt - 1))
            logger.warning('%s failed with HTTP %s, retrying in %.1fs (%d/%d).', label, status, delay, attempt, retries)
            time.sleep(delay)
        except Exception as exc:  # pragma: no cover - defensive around network/auth stack
            last_error = exc
            if attempt >= retries:
                raise
            delay = base_delay_s * (2 ** (attempt - 1))
            logger.warning('%s failed, retrying in %.1fs (%d/%d): %s', label, delay, attempt, retries, exc)
            time.sleep(delay)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f'{label} failed without raising an explicit exception')


def create_youtube_playlist(
    youtube,
    title: str,
    video_ids: list[str],
    description: str = 'Created by Mai (reordered)',
    privacy_status: str = 'unlisted',
    retries: int = 3,
) -> dict[str, Any]:
    """Create a standard YouTube playlist and add videos in order."""
    privacy = str(privacy_status).strip().lower() or 'unlisted'
    if privacy not in {'private', 'public', 'unlisted'}:
        raise ValueError('privacy_status must be one of: private, public, unlisted')

    body = {
        'snippet': {
            'title': title,
            'description': description,
        },
        'status': {
            'privacyStatus': privacy,
        },
    }
    created = _execute_with_retry(
        lambda: youtube.playlists().insert(part='snippet,status', body=body),
        label=f'create playlist "{title}"',
        retries=retries,
    )
    playlist_id = created['id']
    failed_video_ids: list[str] = []

    insert_position = 0
    for video_id in [str(video_id).strip() for video_id in video_ids if str(video_id).strip()]:
        item_body = {
            'snippet': {
                'playlistId': playlist_id,
                'position': insert_position,
                'resourceId': {
                    'kind': 'youtube#video',
                    'videoId': video_id,
                },
            }
        }
        try:
            _execute_with_retry(
                lambda item_body=item_body: youtube.playlistItems().insert(part='snippet', body=item_body),
                label=f'insert video {video_id}',
                retries=retries,
            )
            insert_position += 1
        except Exception as exc:  # pragma: no cover - depends on live API responses
            failed_video_ids.append(video_id)
            logger.warning('Skipping %s during YouTube export: %s', video_id, exc)

    return {
        'playlist_id': playlist_id,
        'playlist_url': f'https://www.youtube.com/playlist?list={playlist_id}',
        'failed_video_ids': failed_video_ids,
    }
