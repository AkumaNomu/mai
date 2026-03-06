import os
import re
from typing import Any

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials


def _norm_text(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"\s+", " ", lowered)
    lowered = re.sub(r"[^a-z0-9 ]", "", lowered)
    return lowered


class SpotifyGenreResolver:
    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        market: str = "US",
    ) -> None:
        self.market = market
        self._artist_cache: dict[str, list[str]] = {}
        self._lookup_cache: dict[str, list[str]] = {}

        cid = client_id or os.getenv("SPOTIFY_CLIENT_ID")
        secret = client_secret or os.getenv("SPOTIFY_CLIENT_SECRET")
        if not cid or not secret:
            self.client = None
            return

        auth_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
        self.client = spotipy.Spotify(auth_manager=auth_manager)

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def _artist_genres(self, artist_id: str) -> list[str]:
        if artist_id in self._artist_cache:
            return self._artist_cache[artist_id]
        if not self.client:
            return []

        try:
            artist = self.client.artist(artist_id)
            genres = artist.get("genres", []) or []
        except Exception:
            genres = []
        self._artist_cache[artist_id] = genres
        return genres

    def _pick_track(self, items: list[dict[str, Any]], title: str, artist: str) -> dict[str, Any] | None:
        if not items:
            return None
        if not artist:
            return items[0]

        title_n = _norm_text(title)
        artist_n = _norm_text(artist)
        best_item = items[0]
        best_score = -1
        for item in items:
            item_title = _norm_text(item.get("name", ""))
            artists = [a.get("name", "") for a in item.get("artists", [])]
            artist_text = " ".join(_norm_text(a) for a in artists)
            score = 0
            if title_n and title_n in item_title:
                score += 2
            if artist_n and artist_n in artist_text:
                score += 3
            if score > best_score:
                best_score = score
                best_item = item
        return best_item

    def lookup_genres(self, title: str, artist: str = "") -> list[str]:
        if not self.client:
            return []
        cache_key = f"{_norm_text(title)}|{_norm_text(artist)}"
        if cache_key in self._lookup_cache:
            return self._lookup_cache[cache_key]

        query = f'track:"{title}"'
        if artist:
            query += f' artist:"{artist}"'

        try:
            result = self.client.search(q=query, type="track", limit=5, market=self.market)
            items = result.get("tracks", {}).get("items", []) or []
        except Exception:
            items = []

        picked = self._pick_track(items, title, artist)
        if not picked:
            self._lookup_cache[cache_key] = []
            return []

        genres: list[str] = []
        for art in picked.get("artists", []):
            artist_id = art.get("id")
            if artist_id:
                genres.extend(self._artist_genres(artist_id))

        out = sorted(set(genres))
        self._lookup_cache[cache_key] = out
        return out
