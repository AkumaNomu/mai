import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import ctypes
import glob
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence
import numpy as np
import pandas as pd
import librosa
from yt_dlp import YoutubeDL

from .sentiment import add_sentiment_features
from .tabular_cache import read_sqlite_table, resolve_sqlite_cache_path, write_sqlite_table
from .yt_dlp_auth import apply_yt_dlp_auth_options, ensure_yt_dlp_ffmpeg_location


logger = logging.getLogger(__name__)
ProgressCallback = Callable[[str, int, int, str], None]
_TEMP_AUDIO_CACHE_SUFFIXES = {'.part', '.tmp', '.temp', '.ytdl'}
_RESOURCE_PROFILE_DEFAULT = 'default'
_RESOURCE_PROFILE_BACKGROUND = 'background'
_RESOURCE_PROFILE_CHOICES = (_RESOURCE_PROFILE_DEFAULT, _RESOURCE_PROFILE_BACKGROUND)
_BACKGROUND_THREAD_ENV_VARS = (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
    'BLIS_NUM_THREADS',
)

_FEATURE_CACHE_VERSION = 1
_FEATURE_CACHE_KEY_COLUMN = 'video_id'
_FEATURE_CACHE_SIGNATURE_COLUMN = 'analysis_signature'
_FEATURE_CACHE_RESERVED_COLUMNS = [
    _FEATURE_CACHE_KEY_COLUMN,
    _FEATURE_CACHE_SIGNATURE_COLUMN,
]
_FEATURE_CACHE_LEGACY_SETTINGS_COLUMNS = [
    'cache_version',
    'edge_seconds',
    'silence_top_db',
    'flow_profile',
]
_FEATURE_CACHE_LEGACY_TEXT_COLUMNS = [
    'title',
    'artist',
    'uploader',
    'channel',
    'description',
    'tags',
    'category',
    'url',
    'context_text',
]
_FEATURE_CACHE_CONTEXT_SOURCE_COLUMNS = [
    'title',
    'artist',
    'uploader',
    'channel',
    'description',
    'tags',
    'category',
]
_DEFAULT_FEATURE_CACHE_DB_PATH = os.path.join('data', 'cache', 'audio_features.sqlite')
_FEATURE_CACHE_TABLE_NAME = 'audio_features'


def _ensure_ffmpeg_dir(cache_root: str) -> Optional[str]:
    return ensure_yt_dlp_ffmpeg_location(cache_root)


def _emit_progress(
    progress_callback: ProgressCallback | None,
    label: str,
    current: int,
    total: int,
    detail: str = '',
) -> None:
    if progress_callback is not None:
        progress_callback(label, int(current), int(total), str(detail or ''))


def _worker_count(requested_workers: int, total_items: int) -> int:
    requested = max(int(requested_workers or 1), 1)
    return max(1, min(requested, max(int(total_items), 1)))


def _normalize_resource_profile(resource_profile: str | None) -> str:
    normalized = str(resource_profile or _RESOURCE_PROFILE_DEFAULT).strip().lower()
    return normalized if normalized in _RESOURCE_PROFILE_CHOICES else _RESOURCE_PROFILE_DEFAULT


def _resolve_analysis_resource_settings(
    *,
    download_workers: int,
    analysis_workers: int,
    resource_profile: str | None,
) -> dict[str, int | str | bool]:
    normalized_profile = _normalize_resource_profile(resource_profile)
    effective_download_workers = max(int(download_workers or 1), 1)
    effective_analysis_workers = max(int(analysis_workers or 1), 1)
    force_process_pool = False
    if normalized_profile == _RESOURCE_PROFILE_BACKGROUND:
        effective_download_workers = 1
        effective_analysis_workers = 1
        force_process_pool = True
    return {
        'resource_profile': normalized_profile,
        'download_workers': effective_download_workers,
        'analysis_workers': effective_analysis_workers,
        'force_process_pool': force_process_pool,
    }


def _prepare_background_worker_environment(resource_profile: str | None) -> None:
    if _normalize_resource_profile(resource_profile) != _RESOURCE_PROFILE_BACKGROUND:
        return
    for env_var in _BACKGROUND_THREAD_ENV_VARS:
        os.environ[env_var] = '1'


def _lower_current_process_priority() -> None:
    if os.name == 'nt':
        try:
            BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.kernel32.SetPriorityClass(handle, BELOW_NORMAL_PRIORITY_CLASS)
        except Exception:
            return
        return
    try:
        os.nice(10)
    except OSError:
        return


def _initialize_analysis_worker(resource_profile: str | None) -> None:
    if _normalize_resource_profile(resource_profile) != _RESOURCE_PROFILE_BACKGROUND:
        return
    _lower_current_process_priority()


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _norm01(value: float, vmin: float, vmax: float) -> float:
    if vmax <= vmin:
        return 0.0
    return _clip01((value - vmin) / (vmax - vmin))


def _serialize_cache_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if pd.isna(value):
        return None
    return value


def _coerce_float(value, default: float = float('nan')) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _analysis_signature(
    edge_seconds: float,
    silence_top_db: float,
    flow_profile: str,
    *,
    cache_version: int = _FEATURE_CACHE_VERSION,
) -> str:
    return (
        f'v{int(cache_version)}'
        f'|edge={float(edge_seconds):g}'
        f'|silence={float(silence_top_db):g}'
        f'|flow={str(flow_profile)}'
    )


def _analysis_signature_from_row(row: pd.Series | dict) -> str:
    value = row.get(_FEATURE_CACHE_SIGNATURE_COLUMN, '')
    if not pd.isna(value):
        text = str(value).strip()
        if text:
            return text
    return _analysis_signature(
        edge_seconds=_coerce_float(row.get('edge_seconds'), default=-1.0),
        silence_top_db=_coerce_float(row.get('silence_top_db'), default=-1.0),
        flow_profile=str(row.get('flow_profile', '')),
        cache_version=int(_coerce_float(row.get('cache_version'), default=-1.0)),
    )


def _feature_cache_lookup_key(video_id: str, analysis_signature: str) -> tuple[str, str]:
    return (str(video_id).strip(), str(analysis_signature).strip())


def _normalize_context_text(*values: object) -> str:
    parts = []
    for value in values:
        if pd.isna(value):
            continue
        text = ' '.join(str(value).strip().split())
        if text:
            parts.append(text)
    if not parts:
        return ''
    return ' | '.join(parts).casefold()


def _metadata_from_row(row: pd.Series, url_col: str = 'url') -> dict:
    metadata = {}
    for column in _FEATURE_CACHE_CONTEXT_SOURCE_COLUMNS:
        value = row.get(column, '')
        metadata[column] = '' if pd.isna(value) else str(value).strip()
    url_value = row.get(url_col, row.get('url', ''))
    metadata['url'] = '' if pd.isna(url_value) else str(url_value).strip()
    metadata['context_text'] = _normalize_context_text(*(metadata.get(column, '') for column in _FEATURE_CACHE_CONTEXT_SOURCE_COLUMNS))
    return metadata


def _merge_metadata_dicts(existing: Optional[dict], new_values: Optional[dict]) -> dict:
    merged = dict(existing or {})
    for key, value in dict(new_values or {}).items():
        text = '' if value is None else str(value).strip()
        if not text:
            continue
        if key == 'context_text':
            if len(text) > len(str(merged.get(key) or '')):
                merged[key] = text
            continue
        if not str(merged.get(key) or '').strip():
            merged[key] = text
    if not str(merged.get('context_text') or '').strip():
        merged['context_text'] = _normalize_context_text(
            *(merged.get(column, '') for column in _FEATURE_CACHE_CONTEXT_SOURCE_COLUMNS)
        )
    return merged


def _task_display_label(video_id: str, metadata: Optional[dict] = None) -> str:
    metadata = dict(metadata or {})
    title = str(metadata.get('title') or '').strip()
    artist = str(metadata.get('artist') or '').strip()
    if title and artist:
        return f'{artist} - {title}'
    if title:
        return title
    if artist:
        return artist
    return str(video_id)


def _task_progress_detail(video_id: str, metadata: Optional[dict] = None) -> str:
    display = _task_display_label(video_id, metadata)
    normalized_video_id = str(video_id).strip()
    if display and display != normalized_video_id:
        return f'{normalized_video_id} | {display}'
    return normalized_video_id


def _augment_feature_payload(features: dict) -> dict:
    feature_df = add_sentiment_features(pd.DataFrame([features]))
    return {key: _serialize_cache_value(value) for key, value in feature_df.iloc[0].to_dict().items()}


def _augment_feature_cache_table(cache_df: pd.DataFrame) -> pd.DataFrame:
    if cache_df.empty:
        return cache_df

    return add_sentiment_features(cache_df.copy())


def _resolve_feature_cache_paths(feature_cache_dir: Optional[str]) -> tuple[str, str, Optional[str]]:
    feature_cache_db_path, legacy_feature_cache_csv_path = resolve_sqlite_cache_path(
        feature_cache_dir,
        default_path=_DEFAULT_FEATURE_CACHE_DB_PATH,
    )
    legacy_feature_cache_dir = os.path.splitext(legacy_feature_cache_csv_path)[0]
    if not legacy_feature_cache_dir or not os.path.isdir(legacy_feature_cache_dir):
        legacy_feature_cache_dir = None
    return feature_cache_db_path, legacy_feature_cache_csv_path, legacy_feature_cache_dir


def _feature_cache_record(
    video_id: str,
    features: dict,
    edge_seconds: float,
    silence_top_db: float,
    flow_profile: str,
    metadata: Optional[dict] = None,
) -> dict:
    record = {
        _FEATURE_CACHE_KEY_COLUMN: str(video_id),
        _FEATURE_CACHE_SIGNATURE_COLUMN: _analysis_signature(
            edge_seconds=edge_seconds,
            silence_top_db=silence_top_db,
            flow_profile=flow_profile,
        ),
    }
    record.update(_augment_feature_payload(features))
    return record


def _prepare_feature_cache_table(cache_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if cache_df is None or cache_df.empty:
        return pd.DataFrame(columns=_FEATURE_CACHE_RESERVED_COLUMNS)

    prepared = cache_df.copy()
    if _FEATURE_CACHE_KEY_COLUMN not in prepared.columns:
        logger.warning('Ignoring invalid audio feature cache table without %s column.', _FEATURE_CACHE_KEY_COLUMN)
        return pd.DataFrame(columns=_FEATURE_CACHE_RESERVED_COLUMNS)

    prepared[_FEATURE_CACHE_KEY_COLUMN] = prepared[_FEATURE_CACHE_KEY_COLUMN].fillna('').astype(str).str.strip()
    prepared = prepared.loc[prepared[_FEATURE_CACHE_KEY_COLUMN] != ''].copy()
    if prepared.empty:
        return pd.DataFrame(columns=_FEATURE_CACHE_RESERVED_COLUMNS)

    if _FEATURE_CACHE_SIGNATURE_COLUMN not in prepared.columns:
        prepared[_FEATURE_CACHE_SIGNATURE_COLUMN] = ''
    missing_signature = prepared[_FEATURE_CACHE_SIGNATURE_COLUMN].fillna('').astype(str).str.strip().eq('')
    if missing_signature.any():
        prepared.loc[missing_signature, _FEATURE_CACHE_SIGNATURE_COLUMN] = prepared.loc[missing_signature].apply(
            _analysis_signature_from_row,
            axis=1,
        )
    prepared[_FEATURE_CACHE_SIGNATURE_COLUMN] = prepared[_FEATURE_CACHE_SIGNATURE_COLUMN].fillna('').astype(str).str.strip()
    prepared = prepared.loc[prepared[_FEATURE_CACHE_SIGNATURE_COLUMN] != ''].copy()
    if prepared.empty:
        return pd.DataFrame(columns=_FEATURE_CACHE_RESERVED_COLUMNS)

    prepared = prepared.drop_duplicates(subset=_FEATURE_CACHE_RESERVED_COLUMNS, keep='last').reset_index(drop=True)
    prepared = _augment_feature_cache_table(prepared)
    drop_columns = [
        column
        for column in (_FEATURE_CACHE_LEGACY_SETTINGS_COLUMNS + _FEATURE_CACHE_LEGACY_TEXT_COLUMNS)
        if column in prepared.columns
    ]
    if drop_columns:
        prepared = prepared.drop(columns=drop_columns)
    ordered_columns = [column for column in _FEATURE_CACHE_RESERVED_COLUMNS if column in prepared.columns]
    ordered_columns.extend(sorted(column for column in prepared.columns if column not in ordered_columns))
    return prepared.reindex(columns=ordered_columns)


def _read_feature_cache_table(feature_cache_path: str) -> pd.DataFrame:
    if not feature_cache_path or not os.path.exists(feature_cache_path):
        return pd.DataFrame(columns=_FEATURE_CACHE_RESERVED_COLUMNS)
    if str(feature_cache_path).lower().endswith(('.sqlite', '.db')):
        try:
            cache_df = read_sqlite_table(
                feature_cache_path,
                columns=_FEATURE_CACHE_RESERVED_COLUMNS,
                table_name=_FEATURE_CACHE_TABLE_NAME,
            )
        except Exception as exc:
            logger.warning('Failed to read audio feature cache DB %s: %r', feature_cache_path, exc)
            return pd.DataFrame(columns=_FEATURE_CACHE_RESERVED_COLUMNS)
        return _prepare_feature_cache_table(cache_df)
    try:
        cache_df = pd.read_csv(
            feature_cache_path,
            dtype={
                _FEATURE_CACHE_KEY_COLUMN: str,
                _FEATURE_CACHE_SIGNATURE_COLUMN: str,
                'flow_profile': str,
            },
            low_memory=False,
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=_FEATURE_CACHE_RESERVED_COLUMNS)
    except Exception as exc:
        logger.warning('Failed to read audio feature cache CSV %s: %r', feature_cache_path, exc)
        return pd.DataFrame(columns=_FEATURE_CACHE_RESERVED_COLUMNS)
    return _prepare_feature_cache_table(cache_df)


def _write_feature_cache_table(cache_df: pd.DataFrame, feature_cache_path: str) -> None:
    if not feature_cache_path:
        return
    prepared = _prepare_feature_cache_table(cache_df)
    if str(feature_cache_path).lower().endswith(('.sqlite', '.db')):
        write_sqlite_table(
            feature_cache_path,
            prepared,
            columns=prepared.columns.tolist() or _FEATURE_CACHE_RESERVED_COLUMNS,
            table_name=_FEATURE_CACHE_TABLE_NAME,
            key_columns=_FEATURE_CACHE_RESERVED_COLUMNS,
        )
        return
    directory = os.path.dirname(feature_cache_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    temp_dir = directory or '.'
    temp_handle, temp_path = tempfile.mkstemp(
        prefix='audio_features_',
        suffix='.tmp',
        dir=temp_dir,
    )
    os.close(temp_handle)
    try:
        prepared.to_csv(temp_path, index=False)
        os.replace(temp_path, feature_cache_path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _load_legacy_feature_cache_rows(legacy_feature_cache_dir: Optional[str]) -> pd.DataFrame:
    if not legacy_feature_cache_dir or not os.path.isdir(legacy_feature_cache_dir):
        return pd.DataFrame(columns=_FEATURE_CACHE_RESERVED_COLUMNS)

    records = []
    for cache_path in sorted(glob.glob(os.path.join(legacy_feature_cache_dir, '*.json'))):
        try:
            with open(cache_path, 'r', encoding='utf-8') as handle:
                payload = json.load(handle)
        except Exception as exc:
            logger.warning('Skipping unreadable legacy audio feature cache file %s: %r', cache_path, exc)
            continue

        features = payload.get('features')
        if not isinstance(features, dict):
            logger.warning('Skipping legacy audio feature cache file without feature payload: %s', cache_path)
            continue

        settings = payload.get('settings', {})
        record = _feature_cache_record(
            video_id=os.path.splitext(os.path.basename(cache_path))[0],
            features=features,
            edge_seconds=_coerce_float(settings.get('edge_seconds'), default=-1.0),
            silence_top_db=_coerce_float(settings.get('silence_top_db'), default=-1.0),
            flow_profile=str(settings.get('flow_profile', '')),
        )
        record[_FEATURE_CACHE_SIGNATURE_COLUMN] = _analysis_signature(
            edge_seconds=_coerce_float(settings.get('edge_seconds'), default=-1.0),
            silence_top_db=_coerce_float(settings.get('silence_top_db'), default=-1.0),
            flow_profile=str(settings.get('flow_profile', '')),
            cache_version=int(payload.get('cache_version', _FEATURE_CACHE_VERSION)),
        )
        records.append(record)

    return _prepare_feature_cache_table(pd.DataFrame(records))


def _merge_missing_legacy_feature_cache_rows(
    cache_df: pd.DataFrame,
    legacy_cache_df: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    prepared_cache_df = _prepare_feature_cache_table(cache_df)
    prepared_legacy_df = _prepare_feature_cache_table(legacy_cache_df)
    if prepared_legacy_df.empty:
        return prepared_cache_df, 0

    existing_ids = (
        set(zip(
            prepared_cache_df[_FEATURE_CACHE_KEY_COLUMN].astype(str),
            prepared_cache_df[_FEATURE_CACHE_SIGNATURE_COLUMN].astype(str),
        ))
        if not prepared_cache_df.empty else set()
    )
    missing_rows = prepared_legacy_df.loc[
        ~prepared_legacy_df.apply(
            lambda row: (
                str(row.get(_FEATURE_CACHE_KEY_COLUMN, '')).strip(),
                str(row.get(_FEATURE_CACHE_SIGNATURE_COLUMN, '')).strip(),
            ) in existing_ids,
            axis=1,
        )
    ].copy()
    if missing_rows.empty:
        return prepared_cache_df, 0
    if prepared_cache_df.empty:
        return _prepare_feature_cache_table(missing_rows), int(len(missing_rows))

    merged = pd.concat([prepared_cache_df, missing_rows], ignore_index=True, sort=False)
    return _prepare_feature_cache_table(merged), int(len(missing_rows))


def _load_feature_cache_table(
    feature_cache_dir: Optional[str],
) -> tuple[pd.DataFrame, str]:
    feature_cache_db_path, legacy_feature_cache_csv_path, legacy_feature_cache_dir = _resolve_feature_cache_paths(feature_cache_dir)
    cache_df = _read_feature_cache_table(feature_cache_db_path)
    if cache_df.empty and legacy_feature_cache_csv_path and os.path.exists(legacy_feature_cache_csv_path):
        cache_df = _read_feature_cache_table(legacy_feature_cache_csv_path)
    legacy_cache_df = _load_legacy_feature_cache_rows(legacy_feature_cache_dir)
    cache_df, imported_rows = _merge_missing_legacy_feature_cache_rows(cache_df, legacy_cache_df)
    should_persist_cache = (
        imported_rows > 0
        or (
            legacy_feature_cache_csv_path
            and legacy_feature_cache_csv_path != feature_cache_db_path
            and os.path.exists(legacy_feature_cache_csv_path)
            and not os.path.exists(feature_cache_db_path)
        )
    )
    if should_persist_cache:
        logger.info(
            'Imported %d legacy audio feature cache row(s) from %s into %s.',
            imported_rows,
            legacy_feature_cache_dir,
            feature_cache_db_path,
        )
        _write_feature_cache_table(cache_df, feature_cache_db_path)
    return cache_df, feature_cache_db_path


def _cached_features_from_row(cache_row: pd.Series) -> dict:
    features = {}
    for key, value in cache_row.items():
        if key in _FEATURE_CACHE_RESERVED_COLUMNS:
            continue
        features[key] = _serialize_cache_value(value)
    return features


def _build_feature_cache_lookup(cache_df: pd.DataFrame) -> dict[tuple[str, str], dict]:
    prepared_cache_df = _prepare_feature_cache_table(cache_df)
    if prepared_cache_df.empty:
        return {}
    return {
        _feature_cache_lookup_key(
            str(row[_FEATURE_CACHE_KEY_COLUMN]),
            str(row[_FEATURE_CACHE_SIGNATURE_COLUMN]),
        ): {key: _serialize_cache_value(value) for key, value in row.items()}
        for row in prepared_cache_df.to_dict(orient='records')
    }


def _lookup_feature_cache_row(
    cache_df: pd.DataFrame | dict[tuple[str, str], dict],
    video_id: str,
    edge_seconds: float,
    silence_top_db: float,
    flow_profile: str,
) -> Optional[dict]:
    if isinstance(cache_df, pd.DataFrame):
        feature_cache_lookup = _build_feature_cache_lookup(cache_df)
    else:
        feature_cache_lookup = cache_df or {}
    cache_row = feature_cache_lookup.get(
        _feature_cache_lookup_key(
            video_id,
            _analysis_signature(
                edge_seconds=edge_seconds,
                silence_top_db=silence_top_db,
                flow_profile=flow_profile,
            ),
        )
    )
    if not cache_row:
        return None
    return _cached_features_from_row(cache_row)


def _upsert_feature_cache_row(
    cache_df: pd.DataFrame,
    feature_cache_csv_path: str,
    video_id: str,
    features: dict,
    edge_seconds: float,
    silence_top_db: float,
    flow_profile: str,
    metadata: Optional[dict] = None,
) -> pd.DataFrame:
    record_df = pd.DataFrame([
        _feature_cache_record(
            video_id=video_id,
            features=features,
            edge_seconds=edge_seconds,
            silence_top_db=silence_top_db,
            flow_profile=flow_profile,
            metadata=metadata,
        )
    ])
    prepared_cache_df = _prepare_feature_cache_table(cache_df)
    if prepared_cache_df.empty:
        updated_cache_df = record_df
    else:
        updated_cache_df = pd.concat([prepared_cache_df, record_df], ignore_index=True, sort=False)
    updated_cache_df = _prepare_feature_cache_table(updated_cache_df)
    _write_feature_cache_table(updated_cache_df, feature_cache_csv_path)
    return updated_cache_df


def _upsert_feature_cache_records(
    cache_df: pd.DataFrame,
    feature_cache_csv_path: str,
    records: list[dict],
) -> pd.DataFrame:
    if not records:
        return _prepare_feature_cache_table(cache_df)
    prepared_cache_df = _prepare_feature_cache_table(cache_df)
    record_df = _prepare_feature_cache_table(pd.DataFrame(records))
    if prepared_cache_df.empty:
        updated_cache_df = record_df
    else:
        combined = pd.concat([prepared_cache_df, record_df], ignore_index=True, sort=False)
        updated_cache_df = combined.drop_duplicates(subset=_FEATURE_CACHE_RESERVED_COLUMNS, keep='last').reset_index(drop=True)
    updated_cache_df = _prepare_feature_cache_table(updated_cache_df)
    _write_feature_cache_table(updated_cache_df, feature_cache_csv_path)
    return updated_cache_df


def _missing_value_mask(series: pd.Series) -> pd.Series:
    missing = series.isna()
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        missing = missing | series.fillna('').astype(str).str.strip().eq('')
    return missing


def _merge_registry_rows(
    df: pd.DataFrame,
    registry_rows: list[dict],
    id_col: str,
) -> pd.DataFrame:
    if not registry_rows:
        return df

    extra_df = pd.DataFrame(registry_rows)
    if extra_df.empty or id_col not in extra_df.columns:
        return df
    extra_df = extra_df.drop_duplicates(subset=[id_col], keep='last').reset_index(drop=True)

    result = df.copy()
    result_ids = result[id_col].fillna('').astype(str)
    lookup_df = extra_df.set_index(id_col)
    for column in lookup_df.columns:
        values = result_ids.map(lookup_df[column])
        if column not in result.columns:
            result[column] = values
            continue
        mask = _missing_value_mask(result[column])
        result.loc[mask, column] = values.loc[mask]
    return result


def _row_has_audio_features(row: pd.Series, flow_profile: str) -> bool:
    required = ['tempo', 'key', 'mode', 'intro_tempo', 'outro_tempo', 'intro_seconds_used', 'outro_seconds_used']
    if flow_profile == 'deep-dj':
        required.extend(['intro_attack_time_s', 'outro_abruptness', 'outro_release_time_s', 'intro_flux_peak'])
    for column in required:
        if column not in row.index or pd.isna(row.get(column)):
            return False
    return True


def _array_norm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    vmax = float(np.max(values)) if values.size else 0.0
    if vmax <= 1e-9:
        return np.zeros_like(values, dtype=float)
    return values / vmax


def _first_index(mask: np.ndarray, default_index: int) -> int:
    indices = np.flatnonzero(mask)
    return int(indices[0]) if indices.size else int(default_index)


def _last_index(mask: np.ndarray, default_index: int) -> int:
    indices = np.flatnonzero(mask)
    return int(indices[-1]) if indices.size else int(default_index)


def _compute_edge_flow_features(
    y: np.ndarray,
    sr: int,
    side: str,
    base_silence_s: float = 0.0,
    profile: str = 'deep-dj'
) -> dict:
    """Compute edge-oriented transition descriptors for an intro or outro segment."""
    if y.size == 0:
        return {}

    hop_length = 512
    frame_length = 2048
    duration_s = float(y.size) / float(sr)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length).flatten()
    if rms.size == 0:
        rms = np.asarray([0.0], dtype=float)
    rms_norm = _array_norm(rms)
    frame_times = librosa.frames_to_time(np.arange(rms.size), sr=sr, hop_length=hop_length)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_norm = _array_norm(onset_env)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        units='frames',
        backtrack=False,
    )
    onset_density = float(len(onset_frames)) / max(duration_s, 1e-6)
    flux_peak = _clip01(float(np.max(onset_norm)) if onset_norm.size else 0.0)

    beat_stability = np.nan
    downbeat_strength = np.nan
    chroma_stability = np.nan
    if profile == 'deep-dj':
        _, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
        beat_frames = np.asarray(beat_frames, dtype=int)
        if beat_frames.size >= 3:
            beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
            beat_intervals = np.diff(beat_times)
            beat_stability = _clip01(1.0 - (float(np.std(beat_intervals)) / max(float(np.mean(beat_intervals)), 1e-6)))
        elif beat_frames.size > 0:
            beat_stability = 0.5
        else:
            beat_stability = 0.0

        if beat_frames.size > 0 and onset_env.size > 0:
            beat_frame = int(beat_frames[0]) if side == 'intro' else int(beat_frames[-1])
            beat_frame = int(np.clip(beat_frame, 0, onset_env.size - 1))
            downbeat_strength = _clip01(float(onset_env[beat_frame]) / max(float(np.max(onset_env)), 1e-6))
        else:
            downbeat_strength = 0.0

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
        if chroma.shape[1] >= 2:
            chroma_frame_norm = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9)
            chroma_diffs = np.linalg.norm(np.diff(chroma_frame_norm, axis=1), axis=0)
            chroma_stability = _clip01(1.0 - float(np.mean(chroma_diffs)) / 1.5)
        else:
            chroma_stability = 1.0

    active_threshold = 0.15
    strong_threshold = 0.80
    tail_energy = float(np.mean(rms_norm[-3:])) if rms_norm.size else 0.0

    if side == 'intro':
        first_active_idx = _first_index(rms_norm >= active_threshold, 0)
        first_strong_idx = _first_index(rms_norm >= strong_threshold, rms_norm.size - 1)
        attack_time_s = float(frame_times[first_strong_idx]) if frame_times.size else 0.0
        rise_slope = _clip01(
            max(float(rms_norm[first_strong_idx]) - float(rms_norm[0]), 0.0) / max(attack_time_s, 0.05)
        )
        pad_silence_s = float(base_silence_s) + (float(frame_times[first_active_idx]) if frame_times.size else 0.0)
        return {
            'intro_attack_time_s': attack_time_s,
            'intro_rise_slope': rise_slope,
            'intro_onset_density': onset_density,
            'intro_flux_peak': flux_peak,
            'intro_beat_stability': beat_stability,
            'intro_pad_silence_s': pad_silence_s,
            'intro_downbeat_strength': downbeat_strength,
            'intro_chroma_stability': chroma_stability,
        }

    last_active_idx = _last_index(rms_norm >= active_threshold, rms_norm.size - 1)
    last_strong_idx = _last_index(rms_norm >= strong_threshold, 0)
    release_time_s = max(duration_s - (float(frame_times[last_strong_idx]) if frame_times.size else 0.0), 0.0)
    pre_tail_energy = float(np.mean(rms_norm[max(0, last_strong_idx - 3):last_strong_idx + 1])) if rms_norm.size else 0.0
    decay_slope = _clip01(max(pre_tail_energy - tail_energy, 0.0) / max(release_time_s, 0.05))
    abruptness = _clip01(
        0.55 * tail_energy + 0.45 * (1.0 - _norm01(release_time_s, 0.0, min(max(duration_s, 1.0), 8.0)))
    )
    tail_silence_s = float(base_silence_s) + max(
        duration_s - (float(frame_times[last_active_idx]) if frame_times.size else duration_s),
        0.0,
    )
    return {
        'outro_release_time_s': release_time_s,
        'outro_decay_slope': decay_slope,
        'outro_abruptness': abruptness,
        'outro_onset_density': onset_density,
        'outro_flux_peak': flux_peak,
        'outro_beat_stability': beat_stability,
        'outro_tail_silence_s': tail_silence_s,
        'outro_downbeat_strength': downbeat_strength,
        'outro_chroma_stability': chroma_stability,
    }


def _estimate_key_kk(chroma_mean: np.ndarray) -> tuple[int, int]:
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    chroma = chroma_mean / (np.linalg.norm(chroma_mean) + 1e-9)
    major_profile = major_profile / np.linalg.norm(major_profile)
    minor_profile = minor_profile / np.linalg.norm(minor_profile)
    best_score = -1.0
    best_key = 0
    best_mode = 1
    for k in range(12):
        score_major = float(np.dot(chroma, np.roll(major_profile, -k)))
        score_minor = float(np.dot(chroma, np.roll(minor_profile, -k)))
        if score_major > best_score:
            best_score = score_major
            best_key = k
            best_mode = 1
        if score_minor > best_score:
            best_score = score_minor
            best_key = k
            best_mode = 0
    return best_key, best_mode


def _compute_features(y: np.ndarray, sr: int, prefix: str = '') -> dict:
    if y.size == 0:
        raise ValueError('audio segment is empty')

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    rms = float(librosa.feature.rms(y=y).mean())
    loudness_db = float(librosa.amplitude_to_db(np.array([rms]), ref=1.0)[0])
    onset_strength = float(librosa.onset.onset_strength(y=y, sr=sr).mean())
    zcr = float(librosa.feature.zero_crossing_rate(y).mean())
    spectral_centroid = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    spectral_bandwidth = float(librosa.feature.spectral_bandwidth(y=y, sr=sr).mean())
    spectral_flatness = float(librosa.feature.spectral_flatness(y=y).mean())
    spectral_rolloff = float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean())

    harmonic, _ = librosa.effects.hpss(y)
    harmonic_ratio = float(np.sum(harmonic ** 2) / (np.sum(y ** 2) + 1e-9))

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    key, mode = _estimate_key_kk(chroma_mean)

    # A few compact timbre descriptors; keep it small but useful for transitions.
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)
    mfcc_mean = mfcc.mean(axis=1)

    tempo_norm = _norm01(float(tempo), 60.0, 180.0)
    onset_norm = _norm01(onset_strength, 0.0, 5.0)
    brightness_norm = _norm01(spectral_centroid, 1000.0, 5000.0)
    energy = _norm01(loudness_db, -60.0, 0.0)
    danceability = _clip01(0.6 * tempo_norm + 0.4 * onset_norm)
    speechiness = _clip01(0.6 * _norm01(spectral_flatness, 0.0, 0.5) + 0.4 * _norm01(zcr, 0.0, 0.2))
    acousticness = _clip01(harmonic_ratio)
    liveness = _norm01(spectral_bandwidth, 500.0, 4000.0)
    valence = _clip01(0.5 * int(mode) + 0.25 * tempo_norm + 0.25 * brightness_norm)

    feats = {
        f'{prefix}tempo': float(tempo),
        f'{prefix}key': int(key),
        f'{prefix}mode': int(mode),
        f'{prefix}loudness': loudness_db,
        f'{prefix}danceability': danceability,
        f'{prefix}energy': energy,
        f'{prefix}speechiness': speechiness,
        f'{prefix}acousticness': acousticness,
        f'{prefix}liveness': liveness,
        f'{prefix}valence': valence,
        f'{prefix}rms': rms,
        f'{prefix}spectral_centroid': spectral_centroid,
        f'{prefix}spectral_bandwidth': spectral_bandwidth,
        f'{prefix}spectral_flatness': spectral_flatness,
        f'{prefix}spectral_rolloff': spectral_rolloff,
        f'{prefix}zcr': zcr,
        f'{prefix}onset_strength': onset_strength,
        f'{prefix}harmonic_ratio': harmonic_ratio,
    }
    for i, v in enumerate(mfcc_mean, start=1):
        feats[f'{prefix}mfcc{i}'] = float(v)
    return feats


def _gather_non_silent(y: np.ndarray, intervals: np.ndarray, target_samples: int, from_start: bool) -> np.ndarray:
    if target_samples <= 0 or y.size == 0:
        return np.asarray([], dtype=y.dtype)
    if intervals.size == 0:
        return y[:min(target_samples, y.size)] if from_start else y[max(0, y.size - target_samples):]

    pieces = []
    remaining = target_samples
    seq = intervals if from_start else intervals[::-1]
    for start, end in seq:
        seg = y[int(start):int(end)]
        if seg.size == 0:
            continue
        if seg.size >= remaining:
            seg = seg[:remaining] if from_start else seg[-remaining:]
            pieces.append(seg)
            remaining = 0
            break
        pieces.append(seg)
        remaining -= seg.size

    if not pieces:
        return np.asarray([], dtype=y.dtype)
    if from_start:
        return np.concatenate(pieces, axis=0)
    pieces = pieces[::-1]
    return np.concatenate(pieces, axis=0)


def analyze_audio_file(
    path: str,
    sr: int = 22050,
    edge_seconds: float = 30.0,
    silence_top_db: float = 35.0,
    flow_profile: str = 'deep-dj'
) -> dict:
    y, sr = librosa.load(path, sr=sr, mono=True)
    if y.size == 0:
        raise ValueError('audio file is empty')

    intervals = librosa.effects.split(y, top_db=silence_top_db)
    leading_silence_s = float(intervals[0][0]) / float(sr) if intervals.size else 0.0
    trailing_silence_s = float(y.size - intervals[-1][1]) / float(sr) if intervals.size else 0.0

    target_samples = int(edge_seconds * sr)
    intro = _gather_non_silent(y, intervals, target_samples=target_samples, from_start=True)
    outro = _gather_non_silent(y, intervals, target_samples=target_samples, from_start=False)

    feats = {}
    feats.update(_compute_features(y, sr, prefix=''))
    if intro.size:
        feats.update(_compute_features(intro, sr, prefix='intro_'))
        feats.update(_compute_edge_flow_features(intro, sr, side='intro', base_silence_s=leading_silence_s, profile=flow_profile))
        feats['intro_seconds_used'] = float(intro.size) / float(sr)
    else:
        feats['intro_seconds_used'] = 0.0
    if outro.size:
        feats.update(_compute_features(outro, sr, prefix='outro_'))
        feats.update(_compute_edge_flow_features(outro, sr, side='outro', base_silence_s=trailing_silence_s, profile=flow_profile))
        feats['outro_seconds_used'] = float(outro.size) / float(sr)
    else:
        feats['outro_seconds_used'] = 0.0

    feats['intro_leading_silence_s'] = leading_silence_s
    feats['outro_trailing_silence_s'] = trailing_silence_s
    return feats


def download_youtube_audio(url: str, video_id: str, audio_cache_dir: str) -> str:
    os.makedirs(audio_cache_dir, exist_ok=True)
    preferred = os.path.join(audio_cache_dir, f'{video_id}.wav')
    if os.path.exists(preferred):
        return preferred
    existing = glob.glob(os.path.join(audio_cache_dir, f'{video_id}.*'))
    # If we have older non-wav cache entries (webm/m4a), re-download and convert to wav.
    for path in existing:
        try:
            if not path.lower().endswith('.wav'):
                os.remove(path)
        except OSError:
            pass

    output_template = os.path.join(audio_cache_dir, f'{video_id}.%(ext)s')
    ffmpeg_location = _ensure_ffmpeg_dir(cache_root=os.path.dirname(audio_cache_dir))
    logger.debug('download_youtube_audio video_id=%s ffmpeg_location=%s', video_id, ffmpeg_location)

    # For analysis we don't need pristine audio; prefer smaller streams and be aggressive about retries.
    # Many "skips" are just transient googlevideo timeouts.
    base_ydl_opts = {
        'format': 'bestaudio[abr<=128]/bestaudio/best',
        'outtmpl': output_template,
        'quiet': True,
        'no_warnings': True,
        'noplaylist': True,
        'noprogress': True,
        'socket_timeout': 60.0,
        'retries': 10,
        'fragment_retries': 10,
        'http_chunk_size': 1024 * 1024,
        'concurrent_fragment_downloads': 1,
        # IPv6 can be flaky in some environments; forcing IPv4 avoids lots of timeouts.
        'source_address': '0.0.0.0',
        # Convert to a SoundFile-friendly format so librosa doesn't need audioread/ffmpeg.
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '0',
        }],
    }
    if ffmpeg_location:
        base_ydl_opts['ffmpeg_location'] = ffmpeg_location

    last_exc: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            # Backoff before retries to avoid hammering the same failing edge.
            if attempt > 1:
                time.sleep(min(30, 2 ** attempt))
            with YoutubeDL(apply_yt_dlp_auth_options(base_ydl_opts)) as ydl:
                ydl.download([url])
            break
        except Exception as exc:
            last_exc = exc
            logger.warning('download attempt %d/3 failed for %s: %r', attempt, video_id, exc)
            continue
    else:
        raise last_exc if last_exc else RuntimeError('unknown yt-dlp download failure')
    if os.path.exists(preferred):
        return preferred
    downloaded = glob.glob(os.path.join(audio_cache_dir, f'{video_id}.*'))
    if not downloaded:
        raise FileNotFoundError(f'no audio downloaded for {video_id}')
    downloaded_sorted = sorted(downloaded, key=lambda p: (0 if p.lower().endswith('.wav') else 1, p))
    return downloaded_sorted[0]


def _download_audio_worker(
    video_id: str,
    url: str,
    audio_cache_dir: str,
) -> str:
    return download_youtube_audio(url, video_id, audio_cache_dir)


def _delete_audio_cache_file(audio_path: str) -> bool:
    normalized_path = str(audio_path or '').strip()
    if not normalized_path:
        return False
    try:
        if os.path.exists(normalized_path):
            os.remove(normalized_path)
            return True
    except OSError as exc:
        logger.debug('failed to delete cached audio %s: %r', normalized_path, exc)
    return False


def _delete_audio_cache_files(audio_paths: list[str]) -> int:
    deleted = 0
    seen_paths: set[str] = set()
    for audio_path in audio_paths:
        normalized_path = str(audio_path or '').strip()
        if not normalized_path or normalized_path in seen_paths:
            continue
        seen_paths.add(normalized_path)
        if _delete_audio_cache_file(normalized_path):
            deleted += 1
    return deleted


def _is_temp_audio_cache_file(audio_path: str | Path) -> bool:
    path = Path(audio_path)
    return bool({suffix.lower() for suffix in path.suffixes} & _TEMP_AUDIO_CACHE_SUFFIXES)


def _audio_cache_video_id(audio_path: str | Path) -> str:
    name = Path(audio_path).name
    return name.split('.', 1)[0].strip() if name else ''


def _preferred_audio_cache_path(audio_paths: list[str]) -> str:
    def _path_sort_key(path: str) -> tuple[int, str]:
        suffix = Path(path).suffix.lower()
        return (0 if suffix == '.wav' else 1, str(path).lower())

    return str(sorted((str(path) for path in audio_paths), key=_path_sort_key)[0])


def analyze_audio_cache_directory(
    audio_cache_dir: str = 'data/audio_cache',
    feature_cache_dir: Optional[str] = None,
    edge_seconds: float = 30.0,
    silence_top_db: float = 35.0,
    flow_profile: str = 'deep-dj',
    resource_profile: str = _RESOURCE_PROFILE_DEFAULT,
    refresh_cache: bool = False,
    analysis_workers: int = 1,
    delete_audio_after_analysis: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> tuple[pd.DataFrame, dict[str, int | str]]:
    resource_settings = _resolve_analysis_resource_settings(
        download_workers=1,
        analysis_workers=analysis_workers,
        resource_profile=resource_profile,
    )
    resource_profile = str(resource_settings['resource_profile'])
    analysis_workers = int(resource_settings['analysis_workers'])
    force_process_pool = bool(resource_settings['force_process_pool'])
    _prepare_background_worker_environment(resource_profile)
    if resource_profile == _RESOURCE_PROFILE_BACKGROUND:
        logger.info('Background audio analysis enabled: using %d low-priority analysis worker.', analysis_workers)

    audio_root = Path(str(audio_cache_dir))
    feature_cache_df, feature_cache_csv_path = _load_feature_cache_table(feature_cache_dir)
    feature_cache_lookup = _build_feature_cache_lookup(feature_cache_df)
    discovered_files = sorted(candidate for candidate in audio_root.iterdir() if candidate.is_file()) if audio_root.exists() else []

    summary: dict[str, int | str] = {
        'audio_files_found': int(len(discovered_files)),
        'unique_tracks_found': 0,
        'temp_files_skipped': 0,
        'invalid_files_skipped': 0,
        'cache_hits': 0,
        'analyzed': 0,
        'failed': 0,
        'audio_files_deleted': 0,
        'audio_files_kept': 0,
        'feature_cache_path': str(feature_cache_csv_path),
    }
    result_rows: list[dict[str, Any]] = []

    grouped_paths: dict[str, list[str]] = {}
    total_files = len(discovered_files)
    _emit_progress(progress_callback, 'Scanning local audio cache', 0, max(total_files, 1), f'checking {total_files} files')
    for file_number, audio_path in enumerate(discovered_files, start=1):
        if _is_temp_audio_cache_file(audio_path):
            summary['temp_files_skipped'] += 1
            _emit_progress(
                progress_callback,
                'Scanning local audio cache',
                file_number,
                max(total_files, 1),
                f'{audio_path.name} temp file skipped',
            )
            continue
        video_id = _audio_cache_video_id(audio_path)
        if not video_id:
            summary['invalid_files_skipped'] += 1
            _emit_progress(
                progress_callback,
                'Scanning local audio cache',
                file_number,
                max(total_files, 1),
                f'{audio_path.name} missing video id',
            )
            continue
        grouped_paths.setdefault(video_id, []).append(str(audio_path))
        _emit_progress(
            progress_callback,
            'Scanning local audio cache',
            file_number,
            max(total_files, 1),
            f'{video_id} discovered in local audio cache',
        )

    summary['unique_tracks_found'] = int(len(grouped_paths))
    pending_tasks: list[dict[str, Any]] = []
    total_tracks = max(int(len(grouped_paths)), 1)
    _emit_progress(progress_callback, 'Checking feature cache', 0, total_tracks, f'checking {len(grouped_paths)} tracks')
    for track_number, (video_id, audio_paths) in enumerate(sorted(grouped_paths.items()), start=1):
        selected_audio_path = _preferred_audio_cache_path(audio_paths)
        if not refresh_cache:
            cached_features = _lookup_feature_cache_row(
                feature_cache_lookup,
                video_id,
                edge_seconds=edge_seconds,
                silence_top_db=silence_top_db,
                flow_profile=flow_profile,
            )
            if cached_features:
                summary['cache_hits'] += 1
                cached_row = dict(cached_features)
                cached_row['video_id'] = video_id
                cached_row['audio_path'] = selected_audio_path
                cached_row['analysis_status'] = 'cached'
                result_rows.append(cached_row)
                if delete_audio_after_analysis:
                    summary['audio_files_deleted'] += _delete_audio_cache_files(audio_paths)
                else:
                    summary['audio_files_kept'] += len(audio_paths)
                _emit_progress(
                    progress_callback,
                    'Checking feature cache',
                    track_number,
                    total_tracks,
                    f'{_task_progress_detail(video_id)} audio cache hit',
                )
                continue
        pending_tasks.append({
            'video_id': video_id,
            'audio_path': selected_audio_path,
            'audio_paths': list(audio_paths),
            'metadata': {'url': f'https://www.youtube.com/watch?v={video_id}'},
        })
        _emit_progress(
            progress_callback,
            'Checking feature cache',
            track_number,
            total_tracks,
            f'{_task_progress_detail(video_id)} queued for analysis',
        )

    if pending_tasks:
        total_pending = len(pending_tasks)
        max_analysis_workers = _worker_count(analysis_workers, total_pending)
        cache_records: list[dict] = []
        _emit_progress(
            progress_callback,
            'Analyzing local audio cache',
            0,
            max(total_pending, 1),
            f'starting {total_pending} tracks with {max_analysis_workers} workers',
        )
        if max_analysis_workers == 1 and not force_process_pool:
            for task_number, task in enumerate(pending_tasks, start=1):
                video_id = str(task['video_id'])
                try:
                    feats = _analyze_audio_worker(
                        str(task['audio_path']),
                        edge_seconds,
                        silence_top_db,
                        flow_profile,
                    )
                except Exception as exc:
                    summary['failed'] += 1
                    logger.warning('skipping local audio %s due to analysis error: %r', video_id, exc)
                    if delete_audio_after_analysis:
                        summary['audio_files_deleted'] += _delete_audio_cache_files(list(task.get('audio_paths') or []))
                    else:
                        summary['audio_files_kept'] += len(list(task.get('audio_paths') or []))
                    result_rows.append({
                        'video_id': video_id,
                        'audio_path': str(task['audio_path']),
                        'analysis_status': 'failed',
                    })
                    progress_detail = f'{_task_progress_detail(video_id, task.get("metadata"))} analysis failed'
                else:
                    cache_record = _feature_cache_record(
                        video_id=video_id,
                        features=feats,
                        edge_seconds=edge_seconds,
                        silence_top_db=silence_top_db,
                        flow_profile=flow_profile,
                        metadata=dict(task.get('metadata') or {}),
                    )
                    cache_records.append(cache_record)
                    feature_cache_lookup[_feature_cache_lookup_key(video_id, str(cache_record[_FEATURE_CACHE_SIGNATURE_COLUMN]))] = cache_record
                    summary['analyzed'] += 1
                    analyzed_row = _cached_features_from_row(cache_record)
                    analyzed_row['video_id'] = video_id
                    analyzed_row['audio_path'] = str(task['audio_path'])
                    analyzed_row['analysis_status'] = 'analyzed'
                    result_rows.append(analyzed_row)
                    if delete_audio_after_analysis:
                        summary['audio_files_deleted'] += _delete_audio_cache_files(list(task.get('audio_paths') or []))
                    else:
                        summary['audio_files_kept'] += len(list(task.get('audio_paths') or []))
                    progress_detail = f'{_task_progress_detail(video_id, task.get("metadata"))} analyzed'
                _emit_progress(
                    progress_callback,
                    'Analyzing local audio cache',
                    task_number,
                    total_pending,
                    progress_detail,
                )
        else:
            future_map = {}
            with ProcessPoolExecutor(
                max_workers=max_analysis_workers,
                initializer=_initialize_analysis_worker,
                initargs=(resource_profile,),
            ) as executor:
                for task in pending_tasks:
                    future = executor.submit(
                        _analyze_audio_worker,
                        str(task['audio_path']),
                        edge_seconds,
                        silence_top_db,
                        flow_profile,
                    )
                    future_map[future] = task
                completed = 0
                for future in as_completed(future_map):
                    completed += 1
                    task = future_map[future]
                    video_id = str(task['video_id'])
                    try:
                        feats = future.result()
                    except Exception as exc:
                        summary['failed'] += 1
                        logger.warning('skipping local audio %s due to analysis error: %r', video_id, exc)
                        result_rows.append({
                            'video_id': video_id,
                            'audio_path': str(task['audio_path']),
                            'analysis_status': 'failed',
                        })
                        progress_detail = f'{_task_progress_detail(video_id, task.get("metadata"))} analysis failed'
                    else:
                        cache_record = _feature_cache_record(
                            video_id=video_id,
                            features=feats,
                            edge_seconds=edge_seconds,
                            silence_top_db=silence_top_db,
                            flow_profile=flow_profile,
                            metadata=dict(task.get('metadata') or {}),
                        )
                        cache_records.append(cache_record)
                        feature_cache_lookup[_feature_cache_lookup_key(video_id, str(cache_record[_FEATURE_CACHE_SIGNATURE_COLUMN]))] = cache_record
                        summary['analyzed'] += 1
                        analyzed_row = _cached_features_from_row(cache_record)
                        analyzed_row['video_id'] = video_id
                        analyzed_row['audio_path'] = str(task['audio_path'])
                        analyzed_row['analysis_status'] = 'analyzed'
                        result_rows.append(analyzed_row)
                        progress_detail = f'{_task_progress_detail(video_id, task.get("metadata"))} analyzed'
                    if delete_audio_after_analysis:
                        summary['audio_files_deleted'] += _delete_audio_cache_files(list(task.get('audio_paths') or []))
                    else:
                        summary['audio_files_kept'] += len(list(task.get('audio_paths') or []))
                    _emit_progress(
                        progress_callback,
                        'Analyzing local audio cache',
                        completed,
                        total_pending,
                        progress_detail,
                    )
        if cache_records:
            feature_cache_df = _upsert_feature_cache_records(
                feature_cache_df,
                feature_cache_csv_path,
                cache_records,
            )

    result_df = pd.DataFrame(result_rows)
    if not result_df.empty and 'video_id' in result_df.columns:
        result_df = result_df.sort_values(['video_id', 'analysis_status'], kind='stable').reset_index(drop=True)
    return result_df, summary


def _analyze_audio_worker(
    audio_path: str,
    edge_seconds: float,
    silence_top_db: float,
    flow_profile: str,
) -> dict:
    return analyze_audio_file(
        audio_path,
        edge_seconds=edge_seconds,
        silence_top_db=silence_top_db,
        flow_profile=flow_profile,
    )


def analyze_youtube_playlist_audio(
    df: pd.DataFrame,
    audio_cache_dir: str = 'data/audio_cache',
    feature_cache_dir: Optional[str] = None,
    id_col: str = 'video_id',
    url_col: str = 'url',
    max_tracks: Optional[int] = None,
    edge_seconds: float = 30.0,
    silence_top_db: float = 35.0,
    flow_profile: str = 'deep-dj',
    resource_profile: str = _RESOURCE_PROFILE_DEFAULT,
    refresh_cache: bool = False,
    download_workers: int = 1,
    analysis_workers: int = 1,
    delete_audio_after_analysis: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    resource_settings = _resolve_analysis_resource_settings(
        download_workers=download_workers,
        analysis_workers=analysis_workers,
        resource_profile=resource_profile,
    )
    resource_profile = str(resource_settings['resource_profile'])
    download_workers = int(resource_settings['download_workers'])
    analysis_workers = int(resource_settings['analysis_workers'])
    force_process_pool = bool(resource_settings['force_process_pool'])
    _prepare_background_worker_environment(resource_profile)
    if resource_profile == _RESOURCE_PROFILE_BACKGROUND:
        logger.info(
            'Background audio analysis enabled: using %d download worker and %d low-priority analysis worker.',
            download_workers,
            analysis_workers,
        )

    registry_rows = []
    feature_cache_df, feature_cache_csv_path = _load_feature_cache_table(feature_cache_dir)
    feature_cache_lookup = _build_feature_cache_lookup(feature_cache_df)
    working_df = df.iloc[:int(max_tracks)].copy() if max_tracks is not None else df.copy()
    total = int(len(working_df))
    analyzed = 0
    reused_from_df = 0
    reused_from_cache = 0
    skipped = 0
    pending_tasks: dict[str, dict[str, Any]] = {}
    _emit_progress(progress_callback, 'Scanning audio cache', 0, max(total, 1), f'checking {total} tracks')

    for row_number, (_, row) in enumerate(working_df.iterrows(), start=1):
        video_id = row.get(id_col)
        if not video_id:
            skipped += 1
            _emit_progress(progress_callback, 'Scanning audio cache', row_number, max(total, 1), 'missing video id')
            continue
        normalized_video_id = str(video_id)
        title = row.get('title', '') or ''
        artist = row.get('artist', '') or ''
        logger.info('[%d/%d] %s%s%s', row_number, total, normalized_video_id, f' | {title}' if title else '', f' | {artist}' if artist else '')
        if _row_has_audio_features(row, flow_profile=flow_profile):
            logger.debug('using existing dataframe features for %s', normalized_video_id)
            reused_from_df += 1
            _emit_progress(
                progress_callback,
                'Scanning audio cache',
                row_number,
                max(total, 1),
                f'{_task_progress_detail(normalized_video_id, _metadata_from_row(row, url_col=url_col))} existing dataframe features',
            )
            continue
        if not refresh_cache:
            cached_features = _lookup_feature_cache_row(
                feature_cache_lookup,
                normalized_video_id,
                edge_seconds=edge_seconds,
                silence_top_db=silence_top_db,
                flow_profile=flow_profile,
            )
            if cached_features:
                logger.debug('using cached feature row for %s from %s', normalized_video_id, feature_cache_csv_path)
                cached_features[id_col] = video_id
                registry_rows.append(cached_features)
                reused_from_cache += 1
                _emit_progress(
                    progress_callback,
                    'Scanning audio cache',
                    row_number,
                    max(total, 1),
                    f'{_task_progress_detail(normalized_video_id, _metadata_from_row(row, url_col=url_col))} audio cache hit',
                )
                continue
        url = str(row.get(url_col) or f'https://www.youtube.com/watch?v={video_id}')
        metadata = _metadata_from_row(row, url_col=url_col)
        task = pending_tasks.get(normalized_video_id)
        if task is None:
            pending_tasks[normalized_video_id] = {
                'video_id': normalized_video_id,
                'url': url,
                'metadata': metadata,
            }
        else:
            task['metadata'] = _merge_metadata_dicts(task.get('metadata'), metadata)
            if not str(task.get('url') or '').strip():
                task['url'] = url
        _emit_progress(
            progress_callback,
            'Scanning audio cache',
            row_number,
            max(total, 1),
            f'{_task_progress_detail(normalized_video_id, metadata)} queued for analysis',
        )

    pending_list = list(pending_tasks.values())
    downloaded_audio_paths: dict[str, str] = {}
    if pending_list:
        total_pending = len(pending_list)
        max_download_workers = _worker_count(download_workers, total_pending)
        _emit_progress(
            progress_callback,
            'Downloading audio',
            0,
            max(total_pending, 1),
            f'starting {total_pending} downloads with {max_download_workers} workers',
        )
        if max_download_workers == 1:
            for task_number, task in enumerate(pending_list, start=1):
                video_id = str(task['video_id'])
                try:
                    downloaded_audio_paths[video_id] = _download_audio_worker(video_id, str(task['url']), audio_cache_dir)
                except Exception as exc:
                    skipped += 1
                    logger.warning('skipping %s due to audio download error: %r', video_id, exc)
                    progress_detail = f'{_task_progress_detail(video_id, task.get("metadata"))} download failed'
                else:
                    progress_detail = f'{_task_progress_detail(video_id, task.get("metadata"))} downloaded'
                _emit_progress(
                    progress_callback,
                    'Downloading audio',
                    task_number,
                    total_pending,
                    progress_detail,
                )
        else:
            future_map = {}
            with ThreadPoolExecutor(max_workers=max_download_workers) as executor:
                for task in pending_list:
                    video_id = str(task['video_id'])
                    future = executor.submit(
                        _download_audio_worker,
                        video_id,
                        str(task['url']),
                        audio_cache_dir,
                    )
                    future_map[future] = task
                completed = 0
                for future in as_completed(future_map):
                    completed += 1
                    task = future_map[future]
                    video_id = str(task['video_id'])
                    try:
                        downloaded_audio_paths[video_id] = future.result()
                    except Exception as exc:
                        skipped += 1
                        logger.warning('skipping %s due to audio download error: %r', video_id, exc)
                        progress_detail = f'{_task_progress_detail(video_id, task.get("metadata"))} download failed'
                    else:
                        progress_detail = f'{_task_progress_detail(video_id, task.get("metadata"))} downloaded'
                    _emit_progress(
                        progress_callback,
                        'Downloading audio',
                        completed,
                        total_pending,
                        progress_detail,
                    )

    analyzed_records: list[dict] = []
    failed_analysis_video_ids: set[str] = set()
    analyzable_tasks = [task for task in pending_list if str(task['video_id']) in downloaded_audio_paths]
    if analyzable_tasks:
        total_analyzable = len(analyzable_tasks)
        max_analysis_workers = _worker_count(analysis_workers, total_analyzable)
        _emit_progress(
            progress_callback,
            'Analyzing audio',
            0,
            max(total_analyzable, 1),
            f'starting {total_analyzable} tracks with {max_analysis_workers} workers',
        )
        if max_analysis_workers == 1 and not force_process_pool:
            for task_number, task in enumerate(analyzable_tasks, start=1):
                video_id = str(task['video_id'])
                try:
                    feats = _analyze_audio_worker(
                        downloaded_audio_paths[video_id],
                        edge_seconds,
                        silence_top_db,
                        flow_profile,
                    )
                except Exception as exc:
                    skipped += 1
                    failed_analysis_video_ids.add(video_id)
                    logger.warning('skipping %s due to audio analysis error: %r', video_id, exc)
                    progress_detail = f'{_task_progress_detail(video_id, task.get("metadata"))} analysis failed'
                else:
                    analyzed_records.append({
                        'video_id': video_id,
                        'metadata': dict(task.get('metadata') or {}),
                        'features': feats,
                    })
                    progress_detail = f'{_task_progress_detail(video_id, task.get("metadata"))} analyzed'
                _emit_progress(
                    progress_callback,
                    'Analyzing audio',
                    task_number,
                    total_analyzable,
                    progress_detail,
                )
        else:
            future_map = {}
            with ProcessPoolExecutor(
                max_workers=max_analysis_workers,
                initializer=_initialize_analysis_worker,
                initargs=(resource_profile,),
            ) as executor:
                for task in analyzable_tasks:
                    video_id = str(task['video_id'])
                    future = executor.submit(
                        _analyze_audio_worker,
                        downloaded_audio_paths[video_id],
                        edge_seconds,
                        silence_top_db,
                        flow_profile,
                    )
                    future_map[future] = task
                completed = 0
                for future in as_completed(future_map):
                    completed += 1
                    task = future_map[future]
                    video_id = str(task['video_id'])
                    try:
                        feats = future.result()
                    except Exception as exc:
                        skipped += 1
                        failed_analysis_video_ids.add(video_id)
                        logger.warning('skipping %s due to audio analysis error: %r', video_id, exc)
                        progress_detail = f'{_task_progress_detail(video_id, task.get("metadata"))} analysis failed'
                    else:
                        analyzed_records.append({
                            'video_id': video_id,
                            'metadata': dict(task.get('metadata') or {}),
                            'features': feats,
                        })
                        progress_detail = f'{_task_progress_detail(video_id, task.get("metadata"))} analyzed'
                    _emit_progress(
                        progress_callback,
                        'Analyzing audio',
                        completed,
                        total_analyzable,
                        progress_detail,
                    )

    if delete_audio_after_analysis and failed_analysis_video_ids:
        for video_id in failed_analysis_video_ids:
            _delete_audio_cache_file(downloaded_audio_paths.get(video_id, ''))

    if analyzed_records:
        cache_records = []
        for record in analyzed_records:
            video_id = str(record['video_id'])
            metadata = dict(record.get('metadata') or {})
            feats = dict(record.get('features') or {})
            cache_record = _feature_cache_record(
                video_id=video_id,
                features=feats,
                edge_seconds=edge_seconds,
                silence_top_db=silence_top_db,
                flow_profile=flow_profile,
                metadata=metadata,
            )
            cache_records.append(cache_record)
            feature_cache_lookup[_feature_cache_lookup_key(video_id, str(cache_record[_FEATURE_CACHE_SIGNATURE_COLUMN]))] = cache_record
            merged_row = _cached_features_from_row(cache_record)
            merged_row[id_col] = video_id
            registry_rows.append(merged_row)
        feature_cache_df = _upsert_feature_cache_records(
            feature_cache_df,
            feature_cache_csv_path,
            cache_records,
        )
        if delete_audio_after_analysis:
            for record in analyzed_records:
                _delete_audio_cache_file(downloaded_audio_paths.get(str(record['video_id']), ''))
        analyzed += len(cache_records)

    if not registry_rows:
        logger.info(
            'Audio analysis complete: analyzed=%d reused_from_df=%d reused_from_cache=%d skipped=%d',
            analyzed,
            reused_from_df,
            reused_from_cache,
            skipped,
        )
        return df
    logger.info(
        'Audio analysis complete: analyzed=%d reused_from_df=%d reused_from_cache=%d skipped=%d',
        analyzed,
        reused_from_df,
        reused_from_cache,
        skipped,
    )
    return _merge_registry_rows(df, registry_rows, id_col=id_col)


def _bootstrap_config(argv: Sequence[str] | None = None) -> tuple[dict, str, bool]:
    from .config import DEFAULT_CONFIG_PATH, load_project_config

    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument('--config', default=DEFAULT_CONFIG_PATH)
    bootstrap.add_argument('--no-config', action='store_true')
    known, _ = bootstrap.parse_known_args(argv)
    config = load_project_config(known.config, use_config=not known.no_config)
    return config, str(known.config), bool(known.no_config)


def _add_bool_override(
    parser: argparse.ArgumentParser,
    *,
    true_flag: str,
    false_flag: str,
    dest: str,
    default: bool,
    true_help: str,
    false_help: str,
) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(true_flag, dest=dest, action='store_true', default=bool(default), help=true_help)
    group.add_argument(false_flag, dest=dest, action='store_false', help=false_help)


def build_audio_cache_parser(config: dict, config_path: str, no_config: bool) -> argparse.ArgumentParser:
    from .config import get_config_value

    parser = argparse.ArgumentParser(description='analyze audio files already present in the local audio cache')
    parser.add_argument('--config', default=config_path, help='Path to project TOML config')
    parser.add_argument('--no-config', action='store_true', default=no_config, help='Ignore the TOML config and use CLI/default values only')
    parser.add_argument('--audio-cache', default=get_config_value(config, 'cache.audio_dir', 'data/audio_cache'), help='Directory containing downloaded audio files to analyze')
    parser.add_argument('--feature-cache', default=os.path.join(get_config_value(config, 'cache.root_dir', 'data/cache'), 'audio_features.sqlite'), help='Global audio feature cache SQLite DB')
    _add_bool_override(
        parser,
        true_flag='--refresh-cache',
        false_flag='--no-refresh-cache',
        dest='refresh_cache',
        default=bool(get_config_value(config, 'analysis.refresh_cache', False)),
        true_help='Ignore reusable feature cache rows and reanalyze local audio files',
        false_help='Reuse feature cache rows even if the config enables refresh mode',
    )
    parser.add_argument('--edge-seconds', type=float, default=float(get_config_value(config, 'analysis.edge_seconds', 30.0)), help='Analyze first/last non-silent seconds')
    parser.add_argument('--silence-top-db', type=float, default=float(get_config_value(config, 'analysis.silence_top_db', 35.0)), help='Silence threshold for trimming (higher trims more)')
    parser.add_argument('--flow-profile', default=get_config_value(config, 'analysis.flow_profile', 'deep-dj'), choices=['standard', 'deep-dj'], help='Transition edge analysis depth')
    parser.add_argument('--resource-profile', default=get_config_value(config, 'analysis.resource_profile', _RESOURCE_PROFILE_DEFAULT), choices=list(_RESOURCE_PROFILE_CHOICES), help='Resource usage profile: `background` throttles audio analysis so it can run more gently in the background')
    parser.add_argument('--analysis-workers', type=int, default=int(get_config_value(config, 'analysis.analysis_workers', 4)), help='Concurrent worker count for CPU-heavy audio feature extraction')
    _add_bool_override(
        parser,
        true_flag='--delete-audio-after-analysis',
        false_flag='--keep-audio-cache',
        dest='delete_audio_after_analysis',
        default=bool(get_config_value(config, 'analysis.delete_audio_after_analysis', True)),
        true_help='Delete local audio files once their features are safely persisted',
        false_help='Keep local audio files after analysis',
    )
    parser.add_argument('--log-level', default=str(config.get('logging', {}).get('level', 'INFO')), choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'], help='Verbosity')
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    from .cli_progress import CliProgressRenderer, configure_cli_logging

    config, config_path, no_config = _bootstrap_config(argv)
    parser = build_audio_cache_parser(config, config_path=config_path, no_config=no_config)
    args = parser.parse_args(argv)

    configure_cli_logging(getattr(logging, str(args.log_level).upper()))
    progress = CliProgressRenderer()
    try:
        progress.section('Analyzing cached audio', detail=str(args.audio_cache))
        result_df, summary = analyze_audio_cache_directory(
            audio_cache_dir=args.audio_cache,
            feature_cache_dir=args.feature_cache,
            edge_seconds=args.edge_seconds,
            silence_top_db=args.silence_top_db,
            flow_profile=args.flow_profile,
            resource_profile=args.resource_profile,
            refresh_cache=bool(args.refresh_cache),
            analysis_workers=int(args.analysis_workers),
            delete_audio_after_analysis=bool(args.delete_audio_after_analysis),
            progress_callback=progress.update,
        )
        progress.success(
            'Audio cache analysis complete',
            detail=(
                f"files={summary['audio_files_found']} "
                f"tracks={summary['unique_tracks_found']} "
                f"cache_hits={summary['cache_hits']} "
                f"analyzed={summary['analyzed']} "
                f"failed={summary['failed']} "
                f"deleted={summary['audio_files_deleted']} "
                f"kept={summary['audio_files_kept']}"
            ),
        )
    finally:
        progress.close()

    print(
        f"Audio cache summary: audio_files_found={summary['audio_files_found']} "
        f"unique_tracks_found={summary['unique_tracks_found']} "
        f"temp_files_skipped={summary['temp_files_skipped']} "
        f"invalid_files_skipped={summary['invalid_files_skipped']} "
        f"cache_hits={summary['cache_hits']} "
        f"analyzed={summary['analyzed']} "
        f"failed={summary['failed']} "
        f"audio_files_deleted={summary['audio_files_deleted']} "
        f"audio_files_kept={summary['audio_files_kept']} "
        f"feature_cache={summary['feature_cache_path']}"
    )
    if not result_df.empty:
        print(f'Rows materialized: {len(result_df)}')


if __name__ == '__main__':
    main()
