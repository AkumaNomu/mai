import logging
from collections import Counter
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import RobustScaler

from .genre import GENRE_SOURCE_MIN_CONFIDENCE, resolve_genres
from .sentiment import SENTIMENT_DIMS, add_sentiment_features
from .transition_model import TransitionModelArtifact, score_transition_matrix as score_transition_model_matrix
from .tonal import kk_key_transition_similarity


logger = logging.getLogger(__name__)
ProgressCallback = Callable[[str, int, int, str], None]

TIMBRE_EDGE_FEATURES = [
    'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5',
    'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
    'harmonic_ratio', 'acousticness',
]
GROOVE_EDGE_FEATURES = ['tempo', 'energy', 'loudness', 'onset_strength', 'danceability']
EDGE_FLOW_REQUIRED = [
    'intro_attack_time_s',
    'intro_rise_slope',
    'intro_flux_peak',
    'outro_release_time_s',
    'outro_abruptness',
    'outro_tail_silence_s',
]
STRUCTURE_REQUIRED = [
    'intro_onset_density',
    'intro_flux_peak',
    'intro_beat_stability',
    'intro_pad_silence_s',
    'intro_downbeat_strength',
    'intro_chroma_stability',
    'outro_onset_density',
    'outro_flux_peak',
    'outro_beat_stability',
    'outro_tail_silence_s',
    'outro_downbeat_strength',
    'outro_chroma_stability',
]
TRANSITION_COMPONENT_WEIGHTS = {
    'edge_flow_score': 0.30,
    'structure_cadence_score': 0.20,
    'timbre_score': 0.20,
    'groove_score': 0.10,
    'sentiment_score': 0.10,
    'tonal_score': 0.10,
}
RESOLVED_GENRE_COLUMNS = ['style_cluster', 'genre_primary', 'genre_confidence', 'genre_source', 'mix_group']
TITLE_CANDIDATE_COLUMNS = ['title', 'track_name', 'name', 'video_title']
ARTIST_CANDIDATE_COLUMNS = ['artist', 'artists', 'channel_title', 'uploader']
TRANSITION_RATING_BANDS = [
    (0.80, 'excellent'),
    (0.65, 'strong'),
    (0.50, 'good'),
    (0.35, 'mixed'),
    (0.00, 'rough'),
]


def _available_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def _emit_progress(
    progress_callback: ProgressCallback | None,
    label: str,
    current: int,
    total: int,
    detail: str = '',
) -> None:
    if progress_callback is not None:
        progress_callback(label, int(current), int(total), str(detail or ''))


def _normalize_group(value) -> str:
    if pd.isna(value):
        return 'unknown'
    text = str(value).strip().lower()
    if not text:
        return 'unknown'
    for separator in [',', ';', '/', '|']:
        if separator in text:
            text = text.split(separator)[0].strip()
    return text or 'unknown'


def _artist_name(value) -> str:
    if pd.isna(value):
        return ''
    return str(value).strip().lower()


def _pick_first_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def _display_label(row: pd.Series, title_column: str | None, artist_column: str | None) -> str:
    title = ''
    artist = ''
    if title_column is not None and title_column in row.index and not pd.isna(row[title_column]):
        title = str(row[title_column]).strip()
    if artist_column is not None and artist_column in row.index and not pd.isna(row[artist_column]):
        artist = str(row[artist_column]).strip()
    if title and artist:
        return f'{artist} - {title}'
    if title:
        return title
    if artist:
        return artist
    if 'video_id' in row.index and not pd.isna(row['video_id']):
        return str(row['video_id']).strip()
    return ''


def _cosine_similarity_01(matrix_a: np.ndarray, matrix_b: np.ndarray | None = None) -> np.ndarray:
    sim = cosine_similarity(matrix_a, matrix_b)
    sim = np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip((sim + 1.0) / 2.0, 0.0, 1.0)


def _scaled_matrix(df: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray | None, list[str]]:
    columns = _available_columns(df, columns)
    if not columns:
        return None, []
    matrix = df[columns].fillna(0).to_numpy(dtype=np.float32)
    scaler = RobustScaler()
    return scaler.fit_transform(matrix), columns


def _cross_closeness(df: pd.DataFrame, from_columns: list[str], to_columns: list[str]) -> np.ndarray | None:
    from_columns = _available_columns(df, from_columns)
    to_columns = _available_columns(df, to_columns)
    if not from_columns or not to_columns or len(from_columns) != len(to_columns):
        return None
    matrix_from = df[from_columns].fillna(0).to_numpy(dtype=np.float32)
    matrix_to = df[to_columns].fillna(0).to_numpy(dtype=np.float32)
    scaler = RobustScaler()
    scaler.fit(np.vstack([matrix_from, matrix_to]))
    matrix_from = scaler.transform(matrix_from)
    matrix_to = scaler.transform(matrix_to)
    diffs = np.abs(matrix_from[:, None, :] - matrix_to[None, :, :]).mean(axis=2)
    return np.exp(-diffs).astype(np.float32)


def _combine_weighted(components: list[tuple[str, np.ndarray, float]]) -> np.ndarray:
    total_weight = sum(weight for _, _, weight in components)
    if total_weight <= 0:
        raise ValueError('no transition components available')
    combined = np.zeros_like(components[0][1], dtype=np.float32)
    for _, component, weight in components:
        combined += weight * component.astype(np.float32)
    combined = combined / float(total_weight)
    np.fill_diagonal(combined, 0.0)
    return np.clip(combined, 0.0, 1.0)


def _adaptive_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors='coerce').fillna(default)


def _adaptive_norm(df: pd.DataFrame, column: str, default: float = 0.0, invert: bool = False) -> np.ndarray:
    values = _adaptive_series(df, column, default=default)
    if values.empty:
        norm = np.zeros(0, dtype=np.float32)
    else:
        lo = float(values.quantile(0.05))
        hi = float(values.quantile(0.95))
        if hi <= lo:
            lo = float(values.min())
            hi = float(values.max())
        if hi <= lo:
            norm = np.full(len(values), float(np.clip(default, 0.0, 1.0)), dtype=np.float32)
        else:
            norm = ((values.to_numpy(dtype=np.float32) - lo) / (hi - lo)).clip(0.0, 1.0)
    if invert:
        norm = 1.0 - norm
    return norm.astype(np.float32)


def _pairwise_similarity(a: np.ndarray, b: np.ndarray, scale: float = 0.25) -> np.ndarray:
    scale = max(scale, 1e-6)
    diffs = np.abs(a[:, None] - b[None, :]) / scale
    return np.exp(-diffs).astype(np.float32)


def transition_score_rating(score: float | None) -> str:
    if score is None or pd.isna(score):
        return 'start'
    for threshold, label in TRANSITION_RATING_BANDS:
        if float(score) >= threshold:
            return label
    return 'rough'


def _path_transition_scores(path: list[int], transition_scores: np.ndarray) -> list[float]:
    if len(path) <= 1:
        return []
    return [float(transition_scores[path[index], path[index + 1]]) for index in range(len(path) - 1)]


def _sort_key_series(df: pd.DataFrame, column: str) -> tuple[pd.Series, pd.Series]:
    values = df[column]
    numeric = pd.to_numeric(values, errors='coerce')
    if numeric.notna().any():
        return numeric.isna(), numeric.fillna(np.inf)
    text = values.fillna('').astype(str).str.lower()
    return pd.Series(False, index=df.index), text


def ordered_playlist_paths_from_dataframe(
    df: pd.DataFrame,
    order_column: str | None = None,
    playlist_column: str | None = None,
) -> list[tuple[str, list[int]]]:
    if df.empty:
        return []

    effective_playlist_column = playlist_column
    if effective_playlist_column is None:
        for candidate in ['playlist_name', 'playlist_index']:
            if candidate in df.columns:
                effective_playlist_column = candidate
                break
    if effective_playlist_column is not None and effective_playlist_column not in df.columns:
        raise ValueError(f'playlist column not found: {effective_playlist_column}')

    effective_order_column = order_column
    if effective_order_column is None and 'position' in df.columns:
        effective_order_column = 'position'
    if effective_order_column is not None and effective_order_column not in df.columns:
        raise ValueError(f'order column not found: {effective_order_column}')

    working = df.reset_index().rename(columns={'index': '_row_index'}).copy()
    sort_columns: list[str] = []
    if effective_playlist_column is not None:
        sort_columns.append(effective_playlist_column)
    if effective_order_column is not None:
        na_key, value_key = _sort_key_series(working, effective_order_column)
        working['_order_key_missing'] = na_key
        working['_order_key_value'] = value_key
        sort_columns.extend(['_order_key_missing', '_order_key_value'])
    sort_columns.append('_row_index')
    working = working.sort_values(sort_columns, kind='stable').reset_index(drop=True)

    if effective_playlist_column is None:
        return [('input_order', working['_row_index'].astype(int).tolist())]

    paths: list[tuple[str, list[int]]] = []
    for group_number, (playlist_value, group) in enumerate(
        working.groupby(effective_playlist_column, sort=False, dropna=False),
        start=1,
    ):
        if pd.isna(playlist_value) or str(playlist_value).strip() == '':
            label = f'playlist_{group_number:02d}'
        else:
            label = str(playlist_value).strip()
        paths.append((label, group['_row_index'].astype(int).tolist()))
    return paths


def build_transition_report(
    df: pd.DataFrame,
    transition_scores: np.ndarray,
    paths: list[list[int]],
    playlist_labels: list[str] | None = None,
    report_name: str = 'recommended_order',
) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame(
            columns=[
                'report_name',
                'playlist_name',
                'from_position',
                'to_position',
                'from_index',
                'to_index',
                'from_label',
                'to_label',
                'transition_score',
                'transition_rating',
                'transition_rank_from_source',
                'remaining_candidates',
                'best_possible_next_label',
                'best_possible_next_score',
                'score_gap_to_best',
            ]
        )

    base_df = df.reset_index(drop=True)
    title_column = _pick_first_column(base_df, TITLE_CANDIDATE_COLUMNS)
    artist_column = _pick_first_column(base_df, ARTIST_CANDIDATE_COLUMNS)
    all_indices = set(range(len(base_df)))
    rows: list[dict[str, Any]] = []

    for playlist_number, path in enumerate(paths, start=1):
        label = playlist_labels[playlist_number - 1] if playlist_labels and playlist_number - 1 < len(playlist_labels) else f'playlist_{playlist_number:02d}'
        if len(path) <= 1:
            continue
        used_indices: set[int] = {path[0]}
        for position, (from_index, to_index) in enumerate(zip(path[:-1], path[1:]), start=1):
            score = float(transition_scores[from_index, to_index])
            candidate_indices = sorted(index for index in all_indices if index not in used_indices)
            if to_index not in candidate_indices:
                candidate_indices.append(to_index)
            candidate_scores = transition_scores[from_index, candidate_indices].astype(np.float32)
            ranked_candidates = sorted(
                zip(candidate_indices, candidate_scores.tolist()),
                key=lambda item: item[1],
                reverse=True,
            )
            transition_rank = next(
                rank for rank, (candidate_index, _) in enumerate(ranked_candidates, start=1)
                if candidate_index == to_index
            )
            best_candidate_index, best_candidate_score = ranked_candidates[0]

            rows.append({
                'report_name': report_name,
                'playlist_name': label,
                'from_position': position,
                'to_position': position + 1,
                'from_index': int(from_index),
                'to_index': int(to_index),
                'from_label': _display_label(base_df.iloc[from_index], title_column, artist_column),
                'to_label': _display_label(base_df.iloc[to_index], title_column, artist_column),
                'transition_score': score,
                'transition_rating': transition_score_rating(score),
                'transition_rank_from_source': int(transition_rank),
                'remaining_candidates': int(len(candidate_indices)),
                'best_possible_next_label': _display_label(base_df.iloc[best_candidate_index], title_column, artist_column),
                'best_possible_next_score': float(best_candidate_score),
                'score_gap_to_best': float(best_candidate_score - score),
            })
            used_indices.add(to_index)

    return pd.DataFrame(rows)


def summarize_transition_report(report_df: pd.DataFrame) -> pd.DataFrame:
    if report_df.empty:
        return pd.DataFrame(
            columns=[
                'report_name',
                'playlist_name',
                'num_transitions',
                'avg_transition_score',
                'min_transition_score',
                'max_transition_score',
                'excellent_share',
                'strong_or_better_share',
                'avg_rank_from_source',
            ]
        )

    summary = (
        report_df.groupby(['report_name', 'playlist_name'], as_index=False)
        .agg(
            num_transitions=('transition_score', 'size'),
            avg_transition_score=('transition_score', 'mean'),
            min_transition_score=('transition_score', 'min'),
            max_transition_score=('transition_score', 'max'),
            avg_rank_from_source=('transition_rank_from_source', 'mean'),
        )
    )
    excellent = (
        report_df.assign(is_excellent=report_df['transition_rating'].eq('excellent'))
        .groupby(['report_name', 'playlist_name'], as_index=False)['is_excellent']
        .mean()
        .rename(columns={'is_excellent': 'excellent_share'})
    )
    strong_or_better = (
        report_df.assign(is_strong_or_better=report_df['transition_score'].ge(0.65))
        .groupby(['report_name', 'playlist_name'], as_index=False)['is_strong_or_better']
        .mean()
        .rename(columns={'is_strong_or_better': 'strong_or_better_share'})
    )
    summary = summary.merge(excellent, on=['report_name', 'playlist_name'], how='left')
    summary = summary.merge(strong_or_better, on=['report_name', 'playlist_name'], how='left')
    return summary


def _edge_flow_component(df: pd.DataFrame) -> np.ndarray | None:
    if not all(column in df.columns for column in EDGE_FLOW_REQUIRED):
        return None

    intro_quick = _adaptive_norm(df, 'intro_attack_time_s', invert=True)
    intro_attack = _adaptive_norm(df, 'intro_attack_time_s')
    intro_rise = _adaptive_norm(df, 'intro_rise_slope')
    intro_flux = _adaptive_norm(df, 'intro_flux_peak')
    intro_hard = np.clip(0.60 * intro_quick + 0.40 * intro_flux, 0.0, 1.0)
    intro_gradual = np.clip(0.55 * intro_attack + 0.30 * (1.0 - intro_rise) + 0.15 * (1.0 - intro_flux), 0.0, 1.0)

    outro_release = _adaptive_norm(df, 'outro_release_time_s')
    outro_abrupt = _adaptive_norm(df, 'outro_abruptness')
    outro_tail = _adaptive_norm(df, 'outro_tail_silence_s')
    outro_downbeat = _adaptive_norm(df, 'outro_downbeat_strength')
    intro_downbeat = _adaptive_norm(df, 'intro_downbeat_strength')
    outro_chroma = _adaptive_norm(df, 'outro_chroma_stability')
    intro_chroma = _adaptive_norm(df, 'intro_chroma_stability')

    abrupt_to_quick = np.outer(outro_abrupt, intro_hard)
    release_to_gradual = np.outer(outro_release, intro_gradual)
    silence_to_hard = np.outer(outro_tail, intro_hard)
    downbeat_lock = np.outer(outro_downbeat, intro_downbeat)
    harmonic_lock = np.outer(outro_chroma, intro_chroma)
    hard_intro_penalty = np.outer(np.clip(1.0 - np.maximum(outro_tail, 0.6 * outro_release), 0.0, 1.0), intro_hard)

    score = (
        0.28 * abrupt_to_quick
        + 0.24 * release_to_gradual
        + 0.18 * silence_to_hard
        + 0.15 * downbeat_lock
        + 0.15 * harmonic_lock
        - 0.12 * hard_intro_penalty
    )
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def _structure_cadence_component(df: pd.DataFrame) -> np.ndarray | None:
    if not all(column in df.columns for column in STRUCTURE_REQUIRED):
        return None

    outro_onset = _adaptive_norm(df, 'outro_onset_density')
    intro_onset = _adaptive_norm(df, 'intro_onset_density')
    outro_flux = _adaptive_norm(df, 'outro_flux_peak')
    intro_flux = _adaptive_norm(df, 'intro_flux_peak')
    outro_beat = _adaptive_norm(df, 'outro_beat_stability')
    intro_beat = _adaptive_norm(df, 'intro_beat_stability')
    outro_tail = _adaptive_norm(df, 'outro_tail_silence_s')
    intro_pad = _adaptive_norm(df, 'intro_pad_silence_s')
    outro_downbeat = _adaptive_norm(df, 'outro_downbeat_strength')
    intro_downbeat = _adaptive_norm(df, 'intro_downbeat_strength')
    outro_chroma = _adaptive_norm(df, 'outro_chroma_stability')
    intro_chroma = _adaptive_norm(df, 'intro_chroma_stability')

    onset_flow = _pairwise_similarity(outro_onset, intro_onset, scale=0.28)
    flux_flow = _pairwise_similarity(outro_flux, intro_flux, scale=0.25)
    beat_flow = _pairwise_similarity(outro_beat, intro_beat, scale=0.22)
    silence_handoff = _pairwise_similarity(outro_tail, intro_pad, scale=0.30)
    downbeat_flow = _pairwise_similarity(outro_downbeat, intro_downbeat, scale=0.20)
    chroma_flow = _pairwise_similarity(outro_chroma, intro_chroma, scale=0.20)

    score = (
        0.20 * onset_flow
        + 0.16 * flux_flow
        + 0.18 * beat_flow
        + 0.14 * silence_handoff
        + 0.16 * downbeat_flow
        + 0.16 * chroma_flow
    )
    return np.clip(score, 0.0, 1.0).astype(np.float32)


def compute_transition_scores(
    df: pd.DataFrame,
    flow_profile: str = 'deep-dj',
    transition_model: TransitionModelArtifact | None = None,
    transition_model_weight: float = 0.0,
    progress_callback: ProgressCallback | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build a directed transition matrix from edge flow, structure, sentiment, groove, timbre, and key features."""
    if transition_model_weight < 0:
        raise ValueError('transition_model_weight must be non-negative')
    model_enabled = transition_model is not None and transition_model_weight > 0
    total_steps = 8 + (1 if model_enabled else 0)
    df = add_sentiment_features(df)
    _emit_progress(progress_callback, 'Transition scoring', 1, total_steps, 'prepared sentiment features')
    components: list[tuple[str, np.ndarray, float]] = []

    edge_flow = _edge_flow_component(df)
    if edge_flow is not None:
        logger.info('Transition scoring: edge-flow handoff component (%s).', flow_profile)
        components.append(('edge_flow_score', edge_flow, TRANSITION_COMPONENT_WEIGHTS['edge_flow_score']))
        _emit_progress(progress_callback, 'Transition scoring', 2, total_steps, 'edge-flow handoff')
    else:
        logger.info('Transition scoring: edge-flow handoff component skipped (missing deep edge columns).')
        _emit_progress(progress_callback, 'Transition scoring', 2, total_steps, 'edge-flow handoff skipped')

    structure_cadence = _structure_cadence_component(df)
    if structure_cadence is not None:
        logger.info('Transition scoring: structure/cadence component.')
        components.append(('structure_cadence_score', structure_cadence, TRANSITION_COMPONENT_WEIGHTS['structure_cadence_score']))
        _emit_progress(progress_callback, 'Transition scoring', 3, total_steps, 'structure/cadence')
    else:
        logger.info('Transition scoring: structure/cadence component skipped (missing deep edge columns).')
        _emit_progress(progress_callback, 'Transition scoring', 3, total_steps, 'structure/cadence skipped')

    outro_timbre = [f'outro_{column}' for column in TIMBRE_EDGE_FEATURES]
    intro_timbre = [f'intro_{column}' for column in TIMBRE_EDGE_FEATURES]
    timbre_flow = _cross_closeness(df, outro_timbre, intro_timbre)
    if timbre_flow is not None:
        logger.info('Transition scoring: outro->intro timbre component.')
        components.append(('timbre_score', timbre_flow, TRANSITION_COMPONENT_WEIGHTS['timbre_score']))
        _emit_progress(progress_callback, 'Transition scoring', 4, total_steps, 'outro->intro timbre')
    else:
        _emit_progress(progress_callback, 'Transition scoring', 4, total_steps, 'outro->intro timbre skipped')

    outro_groove = [f'outro_{column}' for column in GROOVE_EDGE_FEATURES]
    intro_groove = [f'intro_{column}' for column in GROOVE_EDGE_FEATURES]
    groove_flow = _cross_closeness(df, outro_groove, intro_groove)
    if groove_flow is not None:
        logger.info('Transition scoring: groove/energy component.')
        components.append(('groove_score', groove_flow, TRANSITION_COMPONENT_WEIGHTS['groove_score']))
        _emit_progress(progress_callback, 'Transition scoring', 5, total_steps, 'groove/energy')
    else:
        _emit_progress(progress_callback, 'Transition scoring', 5, total_steps, 'groove/energy skipped')

    outro_sentiment = [f'outro_{column}' for column in SENTIMENT_DIMS]
    intro_sentiment = [f'intro_{column}' for column in SENTIMENT_DIMS]
    sentiment_flow = _cross_closeness(df, outro_sentiment, intro_sentiment)
    if sentiment_flow is not None:
        logger.info('Transition scoring: sentiment component.')
        components.append(('sentiment_score', sentiment_flow, TRANSITION_COMPONENT_WEIGHTS['sentiment_score']))
        _emit_progress(progress_callback, 'Transition scoring', 6, total_steps, 'sentiment')
    else:
        _emit_progress(progress_callback, 'Transition scoring', 6, total_steps, 'sentiment skipped')

    key_flow = kk_key_transition_similarity(
        df,
        from_key_col='outro_key',
        from_mode_col='outro_mode',
        to_key_col='intro_key',
        to_mode_col='intro_mode',
    )
    logger.info('Transition scoring: tonal component.')
    components.append(('tonal_score', key_flow.astype(np.float32), TRANSITION_COMPONENT_WEIGHTS['tonal_score']))
    _emit_progress(progress_callback, 'Transition scoring', 7, total_steps, 'tonal')

    if model_enabled:
        try:
            model_flow = score_transition_model_matrix(transition_model, df)
            logger.info('Transition scoring: trained transition model component.')
            components.append(('transition_model_score', model_flow.astype(np.float32), float(transition_model_weight)))
            _emit_progress(progress_callback, 'Transition scoring', 8, total_steps, 'trained transition model')
        except Exception:
            logger.exception('Transition scoring: trained transition model component failed; continuing without it.')
            _emit_progress(progress_callback, 'Transition scoring', 8, total_steps, 'trained transition model skipped')
    elif transition_model_weight > 0:
        logger.info('Transition scoring: trained transition model component skipped (no model provided).')

    transition_scores = _combine_weighted(components)
    _emit_progress(progress_callback, 'Transition scoring', total_steps, total_steps, 'combined transition matrix')
    return transition_scores, df


def ensure_genre_groups(df: pd.DataFrame, genre_column: str | None = None, genre_clusters: int = 8) -> tuple[pd.DataFrame, str]:
    """Backward-compatible wrapper around the richer genre resolver."""
    resolved = resolve_genres(df, genre_column=genre_column, genre_clusters=genre_clusters)
    return resolved, 'mix_group'


def _path_delta(
    last_index: int,
    candidate_index: int,
    state: dict,
    transition_scores: np.ndarray,
    mix_groups: np.ndarray,
    primary_genres: np.ndarray,
    genre_confidences: np.ndarray,
    artists: np.ndarray,
    playlist_size: int,
    global_mix_usage: Counter,
) -> float:
    base_score = float(transition_scores[last_index, candidate_index])
    score = base_score

    candidate_mix = mix_groups[candidate_index]
    last_mix = mix_groups[last_index]
    candidate_artist = artists[candidate_index]
    last_artist = artists[last_index]
    candidate_genre = primary_genres[candidate_index] if genre_confidences[candidate_index] >= GENRE_SOURCE_MIN_CONFIDENCE else 'unknown'
    last_genre = primary_genres[last_index] if genre_confidences[last_index] >= GENRE_SOURCE_MIN_CONFIDENCE else 'unknown'

    if candidate_artist:
        if candidate_artist == last_artist:
            score -= 0.42
        elif candidate_artist in state['artist_counts']:
            score -= 0.14 * state['artist_counts'][candidate_artist]

    if candidate_mix not in state['mix_counts']:
        score += 0.05
    elif candidate_mix == last_mix:
        score -= 0.08

    if candidate_mix != last_mix:
        if base_score >= 0.60:
            score += 0.05
        elif base_score < 0.45:
            score -= 0.12

    projected_share = (state['mix_counts'][candidate_mix] + 1) / float(len(state['path']) + 1)
    if projected_share > 0.45:
        score -= 0.32 * (projected_share - 0.45) / 0.55

    score += 0.04 / (1 + global_mix_usage[candidate_mix])

    if candidate_genre != 'unknown' and candidate_genre not in state['confident_genres']:
        early_factor = max(0.20, 1.0 - (len(state['path']) / max(playlist_size - 1, 1)))
        score += 0.08 * early_factor

    if candidate_genre != 'unknown' and last_genre != 'unknown' and candidate_genre != last_genre:
        if base_score >= 0.58:
            score += 0.05
        elif base_score < 0.45:
            score -= 0.08

    return score


def _choose_seed_scores(
    available_indices: list[int],
    transition_scores: np.ndarray,
    mix_groups: np.ndarray,
    primary_genres: np.ndarray,
    genre_confidences: np.ndarray,
    global_mix_usage: Counter,
) -> list[tuple[float, int]]:
    outgoing = transition_scores[available_indices][:, available_indices]
    quality = outgoing.mean(axis=1) + 0.75 * outgoing.max(axis=1)
    seed_scores = []
    for pos, index in enumerate(available_indices):
        mix_bonus = 0.05 / (1 + global_mix_usage[mix_groups[index]])
        genre_bonus = 0.04 if genre_confidences[index] >= GENRE_SOURCE_MIN_CONFIDENCE and primary_genres[index] != 'unknown' else 0.0
        seed_scores.append((float(quality[pos]) + mix_bonus + genre_bonus, index))
    seed_scores.sort(reverse=True)
    return seed_scores


def _beam_search_playlist(
    available_indices: list[int],
    transition_scores: np.ndarray,
    mix_groups: np.ndarray,
    primary_genres: np.ndarray,
    genre_confidences: np.ndarray,
    artists: np.ndarray,
    playlist_size: int,
    beam_width: int,
    candidate_width: int,
    global_mix_usage: Counter,
    progress_label: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> list[int]:
    if playlist_size <= 0:
        return []
    if len(available_indices) < playlist_size:
        raise ValueError('not enough songs available to fill the requested playlist size')

    seed_scores = _choose_seed_scores(
        available_indices,
        transition_scores,
        mix_groups,
        primary_genres,
        genre_confidences,
        global_mix_usage,
    )
    states = []
    for _, seed in seed_scores[:max(beam_width * 2, 1)]:
        seed_mix = mix_groups[seed]
        seed_artist = artists[seed]
        seed_genre = primary_genres[seed] if genre_confidences[seed] >= GENRE_SOURCE_MIN_CONFIDENCE else 'unknown'
        states.append({
            'path': [seed],
            'score': 0.0,
            'mix_counts': Counter([seed_mix]),
            'artist_counts': Counter([seed_artist]) if seed_artist else Counter(),
            'confident_genres': {seed_genre} if seed_genre != 'unknown' else set(),
        })
    if states:
        _emit_progress(progress_callback, progress_label or 'Playlist generation', 1, playlist_size, 'selected seed track')

    for step_index in range(1, playlist_size):
        next_states = []
        for state in states:
            used = set(state['path'])
            last_index = state['path'][-1]
            candidates = [index for index in available_indices if index not in used]
            if not candidates:
                continue
            ranked = sorted(
                candidates,
                key=lambda candidate: transition_scores[last_index, candidate],
                reverse=True,
            )[:candidate_width]
            for candidate in ranked:
                delta = _path_delta(
                    last_index,
                    candidate,
                    state,
                    transition_scores,
                    mix_groups,
                    primary_genres,
                    genre_confidences,
                    artists,
                    playlist_size,
                    global_mix_usage,
                )
                mix_counts = state['mix_counts'].copy()
                artist_counts = state['artist_counts'].copy()
                confident_genres = set(state['confident_genres'])
                mix_counts[mix_groups[candidate]] += 1
                if artists[candidate]:
                    artist_counts[artists[candidate]] += 1
                if genre_confidences[candidate] >= GENRE_SOURCE_MIN_CONFIDENCE and primary_genres[candidate] != 'unknown':
                    confident_genres.add(primary_genres[candidate])
                next_states.append({
                    'path': state['path'] + [candidate],
                    'score': state['score'] + delta,
                    'mix_counts': mix_counts,
                    'artist_counts': artist_counts,
                    'confident_genres': confident_genres,
                })
        if not next_states:
            raise ValueError(f'playlist generation stalled at position {step_index + 1}')
        next_states.sort(key=lambda state: state['score'], reverse=True)
        states = next_states[:beam_width]
        _emit_progress(
            progress_callback,
            progress_label or 'Playlist generation',
            step_index + 1,
            playlist_size,
            f'selected {step_index + 1}/{playlist_size} tracks',
        )

    best_state = max(states, key=lambda state: state['score'])
    return best_state['path']


def generate_playlist_paths(
    df: pd.DataFrame,
    transition_scores: np.ndarray,
    playlist_size: int,
    num_playlists: int = 1,
    genre_column: str | None = None,
    genre_clusters: int = 8,
    beam_width: int = 8,
    candidate_width: int = 25,
    allow_reuse: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[list[int]], pd.DataFrame, str]:
    """Generate one or more optimized playlists from a large pool."""
    if playlist_size <= 0:
        raise ValueError('playlist_size must be positive')
    if beam_width <= 0 or candidate_width <= 0:
        raise ValueError('beam_width and candidate_width must be positive')

    can_reuse_genres = genre_column is None and all(column in df.columns for column in RESOLVED_GENRE_COLUMNS)
    if can_reuse_genres and not df[RESOLVED_GENRE_COLUMNS].isna().any().any():
        logger.info('Genre resolution: using existing resolved genre columns from the dataframe.')
        df = df.copy()
    else:
        df = resolve_genres(df, genre_column=genre_column, genre_clusters=genre_clusters)
    mix_groups = df['mix_group'].fillna('unknown').map(_normalize_group).to_numpy()
    primary_genres = df['genre_primary'].fillna('unknown').map(_normalize_group).to_numpy()
    genre_confidences = pd.to_numeric(df['genre_confidence'], errors='coerce').fillna(0.0).to_numpy(dtype=float)
    artists = df['artist'].map(_artist_name).to_numpy() if 'artist' in df.columns else np.asarray([''] * len(df))
    available_indices = list(range(len(df)))
    if not allow_reuse and playlist_size * num_playlists > len(available_indices):
        raise ValueError('not enough unique songs available for the requested number of playlists')

    confident_count = int((genre_confidences >= GENRE_SOURCE_MIN_CONFIDENCE).sum())
    logger.info(
        'Genre resolution: %d/%d tracks have confident canonical genres; mixing fallback is style_cluster for the rest.',
        confident_count,
        len(df),
    )

    paths: list[list[int]] = []
    global_mix_usage: Counter = Counter()
    global_used: set[int] = set()
    for playlist_index in range(num_playlists):
        candidate_pool = available_indices if allow_reuse else [index for index in available_indices if index not in global_used]
        logger.info(
            'Generating playlist %d/%d from %d available songs.',
            playlist_index + 1,
            num_playlists,
            len(candidate_pool),
        )
        progress_label = f'Playlist {playlist_index + 1}/{num_playlists}'
        _emit_progress(progress_callback, progress_label, 0, playlist_size, 'starting search')
        path = _beam_search_playlist(
            candidate_pool,
            transition_scores,
            mix_groups,
            primary_genres,
            genre_confidences,
            artists,
            playlist_size=playlist_size,
            beam_width=beam_width,
            candidate_width=candidate_width,
            global_mix_usage=global_mix_usage,
            progress_label=progress_label,
            progress_callback=progress_callback,
        )
        paths.append(path)
        global_used.update(path)
        global_mix_usage.update(mix_groups[index] for index in path)

    return paths, df, 'mix_group'


def playlists_to_dataframe(
    df: pd.DataFrame,
    paths: list[list[int]],
    transition_scores: np.ndarray | None = None,
) -> pd.DataFrame:
    base_df = df.reset_index(drop=True)
    frames = []
    for playlist_number, path in enumerate(paths, start=1):
        playlist_df = base_df.iloc[path].reset_index(drop=True).copy()
        for column in ['playlist_index', 'position', 'playlist_name']:
            if column in playlist_df.columns:
                playlist_df = playlist_df.drop(columns=column)
        playlist_df.insert(0, 'playlist_name', f'playlist_{playlist_number:02d}')
        playlist_df.insert(0, 'position', np.arange(1, len(playlist_df) + 1))
        playlist_df.insert(0, 'playlist_index', playlist_number)
        if transition_scores is not None:
            previous_scores = [np.nan]
            previous_scores.extend(_path_transition_scores(path, transition_scores))
            previous_ratings = [transition_score_rating(score) for score in previous_scores]
            playlist_df.insert(3, 'transition_score_from_previous', previous_scores)
            playlist_df.insert(4, 'transition_rating_from_previous', previous_ratings)
        frames.append(playlist_df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=df.columns)
