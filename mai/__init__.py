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


from .scene_context import analyze_scene_text  # lexicon-only, no heavy deps


def analyze_scene_image(*args, **kwargs):
    from .scene_features import analyze_scene_image as _analyze_scene_image
    return _analyze_scene_image(*args, **kwargs)


def build_scene_target(*args, **kwargs):
    from .scene_match import build_scene_target as _build_scene_target
    return _build_scene_target(*args, **kwargs)


def score_library_against_scene(*args, **kwargs):
    from .scene_match import score_library_against_scene as _score_library_against_scene
    return _score_library_against_scene(*args, **kwargs)


def build_scene_index(*args, **kwargs):
    from .scene_index import build_scene_index as _build_scene_index
    return _build_scene_index(*args, **kwargs)


def run_benchmark(*args, **kwargs):
    from .scene_eval import run_benchmark as _run_benchmark
    return _run_benchmark(*args, **kwargs)


def compare_retrievers(*args, **kwargs):
    from .scene_eval import compare_retrievers as _compare_retrievers
    return _compare_retrievers(*args, **kwargs)


def fit_affect_probe(*args, **kwargs):
    from .affect_probe import fit_affect_probe as _fit_affect_probe
    return _fit_affect_probe(*args, **kwargs)


def load_benchmark(*args, **kwargs):
    from .scene_dataset import load_benchmark as _load_benchmark
    return _load_benchmark(*args, **kwargs)


def ingest_cue_sheets(*args, **kwargs):
    from .scene_dataset import ingest_cue_sheets as _ingest_cue_sheets
    return _ingest_cue_sheets(*args, **kwargs)


def query_scene_index(*args, **kwargs):
    from .scene_index import query_scene_index as _query_scene_index
    return _query_scene_index(*args, **kwargs)


def paired_bootstrap_test(*args, **kwargs):
    from .scene_eval import paired_bootstrap_test as _paired_bootstrap_test
    return _paired_bootstrap_test(*args, **kwargs)


def scene_to_prompt(*args, **kwargs):
    from .scene_generation import scene_to_prompt as _scene_to_prompt
    return _scene_to_prompt(*args, **kwargs)


def plan_mix(*args, **kwargs):
    from .overlay import plan_mix as _plan_mix
    return _plan_mix(*args, **kwargs)


def best_overlay(*args, **kwargs):
    from .overlay import best_overlay as _best_overlay
    return _best_overlay(*args, **kwargs)


def overlay_score(*args, **kwargs):
    from .overlay import overlay_score as _overlay_score
    return _overlay_score(*args, **kwargs)


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
    'load_youtube_service', 'create_youtube_playlist',
    'analyze_scene_text', 'analyze_scene_image', 'build_scene_target',
    'score_library_against_scene', 'build_scene_index', 'run_benchmark',
    'compare_retrievers', 'fit_affect_probe', 'load_benchmark', 'ingest_cue_sheets',
    'query_scene_index', 'paired_bootstrap_test', 'scene_to_prompt',
    'plan_mix', 'best_overlay', 'overlay_score',
]
