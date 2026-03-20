import logging
import os
from typing import Optional
import pandas as pd
from yt_dlp import YoutubeDL

from .yt_dlp_auth import apply_yt_dlp_auth_options


logger = logging.getLogger(__name__)


def _extract_source_playlist_title(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    source_title = ''
    if 'source_playlist_title' in prepared.columns:
        titles = prepared['source_playlist_title'].fillna('').astype(str).str.strip()
        non_empty = titles.loc[titles.ne('')]
        source_title = str(non_empty.iloc[0]) if not non_empty.empty else ''
        prepared = prepared.drop(columns=['source_playlist_title'])
    prepared.attrs['source_playlist_title'] = source_title
    return prepared


def parse_youtube_playlist_id(url_or_id: str) -> str:
    """Extract a YouTube playlist ID from a URL or return the input if already an ID."""
    if not url_or_id:
        raise ValueError('playlist id is required')
    if 'list=' in url_or_id:
        return url_or_id.split('list=')[1].split('&')[0]
    return url_or_id.strip()


def _format_duration(seconds: Optional[int]) -> str:
    if seconds is None:
        return ''
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f'{h}:{m:02d}:{s:02d}'
    return f'{m}:{s:02d}'


def _metadata_cache_path(cache_dir: str, playlist_id: str, limit: Optional[int]) -> str:
    limit_label = 'all' if limit is None else str(int(limit))
    safe_playlist_id = ''.join(ch for ch in playlist_id if ch.isalnum() or ch in ('-', '_')) or 'playlist'
    return os.path.join(cache_dir, f'{safe_playlist_id}_{limit_label}.csv')


def fetch_youtube_playlist_tracks(
    playlist_id: str,
    limit: Optional[int] = None,
    cache_dir: Optional[str] = 'data/cache/youtube_playlists',
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch tracks from a public YouTube playlist via yt-dlp (metadata only)."""
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = _metadata_cache_path(cache_dir, playlist_id, limit)
        if not refresh and os.path.exists(cache_path):
            logger.info('Using cached YouTube playlist metadata: %s', cache_path)
            return _extract_source_playlist_title(pd.read_csv(cache_path))

    playlist_url = f'https://www.youtube.com/playlist?list={playlist_id}'
    ydl_opts = {
        'quiet': True,
        'extract_flat': 'in_playlist',
        'skip_download': True,
    }
    with YoutubeDL(apply_yt_dlp_auth_options(ydl_opts)) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
    source_playlist_title = str((info or {}).get('title') or (info or {}).get('playlist_title') or '').strip()
    tracks = info.get('entries', []) if info else []
    rows = []
    for track in tracks[:limit] if limit else tracks:
        video_id = track.get('id')
        if not video_id:
            continue
        title = track.get('title', '') or ''
        artist_name = track.get('uploader', '') or ''
        duration_seconds = track.get('duration')
        channel_name = track.get('channel') or track.get('uploader') or ''
        tags = track.get('tags') or []
        category = track.get('categories') or track.get('category') or []
        rows.append({
            'title': title,
            'artist': artist_name,
            'video_id': video_id,
            'duration': _format_duration(duration_seconds),
            'duration_seconds': duration_seconds or 0,
            'uploader': artist_name,
            'channel': channel_name,
            'channel_id': track.get('channel_id') or track.get('uploader_id') or '',
            'description': track.get('description') or '',
            'tags': ', '.join(tags) if isinstance(tags, list) else str(tags or ''),
            'category': ', '.join(category) if isinstance(category, list) else str(category or ''),
            'url': f'https://www.youtube.com/watch?v={video_id}&list={playlist_id}',
            'source_playlist_title': source_playlist_title,
        })
    df = pd.DataFrame(rows)
    if cache_path:
        logger.info('Caching YouTube playlist metadata: %s', cache_path)
        df.to_csv(cache_path, index=False)
    return _extract_source_playlist_title(df)
