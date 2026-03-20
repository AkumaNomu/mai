"""mai package: helpers for playlist reordering"""
from .data import load_csv_playlist, ensure_audio_columns, normalize_audio_feature_columns
from .features import add_log_tempo, scale_and_pca
from .tonal import kk_key_similarity, kk_key_transition_similarity
from .similarity import compute_mood_similarity, combine_similarities, sparsify_knn
from .routing import build_graph_from_edges, mst_dfs_tour, two_opt_improve
from .sentiment import add_sentiment_features
from .genre import resolve_genres
from .playlist_generation import (
    build_transition_report,
    compute_transition_scores,
    ensure_genre_groups,
    generate_playlist_paths,
    ordered_playlist_paths_from_dataframe,
    playlists_to_dataframe,
    summarize_transition_report,
    transition_score_rating,
)


def clean_useless_cache(*args, **kwargs):
    from .cache_cleanup import clean_useless_cache as _clean_useless_cache
    return _clean_useless_cache(*args, **kwargs)


try:
    from .youtube_integration import parse_youtube_playlist_id, fetch_youtube_playlist_tracks
except ModuleNotFoundError:  # Optional until YouTube/audio deps are installed.
    parse_youtube_playlist_id = None
    fetch_youtube_playlist_tracks = None


def analyze_audio_file(*args, **kwargs):
    from .audio_analysis import analyze_audio_file as _analyze_audio_file
    return _analyze_audio_file(*args, **kwargs)


def analyze_youtube_playlist_audio(*args, **kwargs):
    from .audio_analysis import analyze_youtube_playlist_audio as _analyze_youtube_playlist_audio
    return _analyze_youtube_playlist_audio(*args, **kwargs)
try:
    from .youtube_export import load_youtube_service, create_youtube_playlist
except ModuleNotFoundError:  # Optional until standard YouTube export deps are installed.
    load_youtube_service = None
    create_youtube_playlist = None

__all__ = [
    'load_csv_playlist', 'ensure_audio_columns', 'normalize_audio_feature_columns',
    'add_log_tempo', 'scale_and_pca', 'kk_key_similarity', 'kk_key_transition_similarity',
    'compute_mood_similarity', 'combine_similarities', 'sparsify_knn',
    'build_graph_from_edges', 'mst_dfs_tour', 'two_opt_improve',
    'clean_useless_cache',
    'parse_youtube_playlist_id', 'fetch_youtube_playlist_tracks',
    'analyze_audio_file', 'analyze_youtube_playlist_audio',
    'add_sentiment_features', 'resolve_genres', 'compute_transition_scores',
    'generate_playlist_paths', 'playlists_to_dataframe', 'ensure_genre_groups',
    'ordered_playlist_paths_from_dataframe', 'build_transition_report',
    'summarize_transition_report', 'transition_score_rating',
    'load_youtube_service', 'create_youtube_playlist'
]
