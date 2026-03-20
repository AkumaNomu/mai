from __future__ import annotations

from typing import Optional
from ytmusicapi import YTMusic


def load_ytmusic(auth_path: str) -> YTMusic:
    """Load an authenticated YTMusic client from a headers auth file."""
    return YTMusic(auth_path)


def create_reordered_playlist(
    ytm: YTMusic,
    title: str,
    video_ids: list[str],
    description: str = 'Created by Mai (reordered)',
    privacy_status: str = 'PRIVATE'
) -> str:
    """Create a YouTube Music playlist and add video IDs in order. Returns the playlist id."""
    playlist_id = ytm.create_playlist(title, description, privacy_status=privacy_status)
    # Add in batches (API limits)
    batch_size = 50
    for i in range(0, len(video_ids), batch_size):
        batch = [vid for vid in video_ids[i:i + batch_size] if vid]
        if not batch:
            continue
        ytm.add_playlist_items(playlist_id, videoIds=batch)
    return playlist_id

