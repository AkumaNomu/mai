from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    tomllib = None


DEFAULT_CONFIG_PATH = 'mai.toml'
DEFAULT_CONFIG: dict[str, Any] = {
    'cache': {
        'root_dir': 'data/cache',
        'audio_dir': 'data/audio_cache',
    },
    'analysis': {
        'refresh_cache': False,
        'edge_seconds': 30.0,
        'silence_top_db': 35.0,
        'flow_profile': 'deep-dj',
        'download_workers': 4,
        'analysis_workers': 4,
        'delete_audio_after_analysis': True,
        'max_tracks': None,
        'no_audio_analysis': False,
    },
    'generation': {
        'playlist_size': None,
        'num_playlists': 1,
        'allow_reuse': False,
        'genre_column': '',
        'genre_clusters': 8,
        'beam_width': 8,
        'candidate_width': 25,
        'input_order_column': '',
        'rate_transitions': False,
        'transition_report_out': '',
        'print_recommended_order': False,
    },
    'training': {
        'output_path': 'data/training/positive_transitions.csv',
        'max_videos': None,
        'max_search_results': 5,
        'metadata_workers': 4,
        'search_workers': 4,
        'label': 'excellent',
        'label_source': 'youtube_mix_curation',
        'videos': [],
        'channels': [
            {
                'name': 'mai_dq',
                'url': 'https://www.youtube.com/@mai_dq/videos',
                'label': 'excellent',
                'label_source': 'mai_dq_mix_curation',
            },
        ],
    },
    'exports': {
        'ytmusic': {
            'auth_path': 'data/ytmusic_auth.json',
            'title': 'mai reordered playlist',
            'privacy': 'PRIVATE',
        },
        'youtube': {
            'client_secrets_path': 'data/youtube_client_secret.json',
            'token_path': 'data/youtube_token.json',
            'title': 'auto',
            'privacy': 'unlisted',
        },
    },
    'logging': {
        'level': 'INFO',
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def get_config_value(config: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = config
    for part in str(path).split('.'):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def load_project_config(
    config_path: str = DEFAULT_CONFIG_PATH,
    use_config: bool = True,
) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if not use_config:
        return config

    path = Path(config_path)
    if not path.exists():
        return config
    if tomllib is None:
        raise RuntimeError('TOML config requires Python 3.11 or newer')

    with path.open('rb') as handle:
        loaded = tomllib.load(handle)
    if not isinstance(loaded, dict):
        return config
    return _deep_merge(config, loaded)
