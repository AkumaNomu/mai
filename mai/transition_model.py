from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .sentiment import add_sentiment_features


logger = logging.getLogger(__name__)

DEFAULT_TRANSITION_MODEL_PATH = os.path.join('data', 'cache', 'transition_model.joblib')
DEFAULT_TRANSITION_MODEL_NEGATIVE_RATIO = 1.0
DEFAULT_TRANSITION_MODEL_RANDOM_STATE = 42
DEFAULT_TRANSITION_MODEL_DEVICE = 'cuda'
MODEL_TARGET_COLUMN = '__transition_target__'

_TIMESTAMP_RE = re.compile(r'^\s*(?:(\d+):)?([0-5]?\d):([0-5]\d)\s*$')
_EXCLUDED_BASE_COLUMNS = {
    'video_id',
    'channel_url',
    'channel_handle',
    'video_title',
    'video_url',
    'description_length',
    'position',
    'timestamp',
    'timestamp_s',
    'track_raw',
    'artist_guess',
    'title_guess',
    'track_source',
    'chapter_title',
    'chapter_timestamp_s',
    'source_signature',
    'resolution_signature',
    'source_cache_version',
    'resolution_cache_version',
    'search_query',
    'normalized_search_query',
    'search_max_results',
    'resolution_status',
    'resolved_video_id',
    'resolved_url',
}
_TEXT_BASE_ALIASES = {
    'resolved_title': ['resolved_title', 'title', 'track_name', 'name', 'video_title'],
    'resolved_artist': ['resolved_artist', 'artist', 'artists', 'channel_title', 'uploader'],
    'genre_primary': ['genre_primary', 'genre', 'genres'],
    'mix_group': ['mix_group', 'genre_primary', 'genre', 'genres'],
    'genre_source': ['genre_source'],
    'style_cluster': ['style_cluster'],
}
_NUMERIC_BASE_HINTS = {
    'tempo',
    'key',
    'mode',
    'loudness',
    'rms',
    'spectral_centroid',
    'spectral_bandwidth',
    'spectral_flatness',
    'spectral_rolloff',
    'zcr',
    'onset_strength',
    'harmonic_ratio',
    'acousticness',
    'danceability',
    'energy',
    'speechiness',
    'liveness',
    'valence',
    'intro_seconds_used',
    'outro_seconds_used',
    'intro_leading_silence_s',
    'outro_trailing_silence_s',
    'intro_attack_time_s',
    'intro_rise_slope',
    'intro_onset_density',
    'intro_flux_peak',
    'intro_beat_stability',
    'intro_pad_silence_s',
    'intro_downbeat_strength',
    'intro_chroma_stability',
    'outro_release_time_s',
    'outro_decay_slope',
    'outro_abruptness',
    'outro_onset_density',
    'outro_flux_peak',
    'outro_beat_stability',
    'outro_tail_silence_s',
    'outro_downbeat_strength',
    'outro_chroma_stability',
    'sentiment_valence',
    'sentiment_arousal',
    'sentiment_tension',
    'sentiment_warmth',
    'genre_confidence',
}
_TORCH_HIDDEN_DIMS = (128, 32)
_TORCH_EPOCHS = 80
_TORCH_BATCH_SIZE = 256


@dataclass(slots=True)
class TransitionFeatureSpec:
    base_columns: list[str]
    numeric_columns: list[str]
    text_columns: list[str]
    feature_names: list[str]


@dataclass(slots=True)
class TorchTransitionModelPayload:
    input_dim: int
    hidden_dims: tuple[int, int]
    state_dict: dict[str, np.ndarray]
    trained_on_device: str = 'cuda'


@dataclass(slots=True)
class TransitionModelArtifact:
    pipeline: Any
    feature_spec: TransitionFeatureSpec
    training_summary: dict[str, Any] = field(default_factory=dict)
    backend: str = 'sklearn'
    device: str = 'cpu'

    def score_transition_matrix(self, df: pd.DataFrame) -> np.ndarray:
        return score_transition_matrix(self, df)


def _normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ''
    return ' '.join(str(value).strip().split())


def _timestamp_to_seconds(text: str) -> float | None:
    match = _TIMESTAMP_RE.match(_normalize_text(text))
    if not match:
        return None
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    return float(hours * 3600 + minutes * 60 + seconds)


def _column_candidates(base_column: str) -> list[str]:
    candidates = list(_TEXT_BASE_ALIASES.get(base_column, []))
    if base_column not in candidates:
        candidates.insert(0, base_column)
    return candidates


def _series_from_candidates(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for column in candidates:
        if column in df.columns:
            return df[column]
    return pd.Series(index=df.index, dtype=object)


def _numeric_series(series: pd.Series, column_name: str) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=float)

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors='coerce')

    values = series.astype(str).map(_normalize_text)
    if 'timestamp' in column_name or column_name.endswith('_s') or column_name.endswith('_seconds'):
        parsed = values.map(_timestamp_to_seconds)
        parsed_series = pd.to_numeric(parsed, errors='coerce')
        if parsed_series.notna().any():
            return parsed_series

    return pd.to_numeric(series, errors='coerce')


def _text_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=object)
    return series.fillna('').astype(str).map(_normalize_text)


def _looks_numeric(base_column: str, series_from: pd.Series, series_to: pd.Series) -> bool:
    if base_column in _NUMERIC_BASE_HINTS:
        return True
    if 'timestamp' in base_column or base_column.endswith('_s') or base_column.endswith('_seconds'):
        return True

    combined = pd.concat([series_from, series_to], ignore_index=True)
    if combined.empty:
        return False

    converted = _numeric_series(combined, base_column)
    if combined.notna().sum() == 0:
        return False
    return float(converted.notna().sum()) / float(combined.notna().sum()) >= 0.6


def _base_transition_columns(df: pd.DataFrame) -> list[str]:
    base_columns = []
    for column in df.columns:
        if not column.startswith('from_'):
            continue
        base = column[5:]
        if f'to_{base}' not in df.columns:
            continue
        if base in _EXCLUDED_BASE_COLUMNS:
            continue
        base_columns.append(base)
    return sorted(dict.fromkeys(base_columns))


def _feature_spec_from_transition_rows(df: pd.DataFrame) -> TransitionFeatureSpec:
    base_columns = _base_transition_columns(df)
    numeric_columns: list[str] = []
    text_columns: list[str] = []
    feature_names: list[str] = []

    for base in base_columns:
        from_series = df[f'from_{base}']
        to_series = df[f'to_{base}']
        if _looks_numeric(base, from_series, to_series):
            numeric_columns.append(base)
            feature_names.extend([
                f'from_{base}',
                f'to_{base}',
                f'delta_{base}',
                f'abs_delta_{base}',
                f'mean_{base}',
            ])
        else:
            text_columns.append(base)
            feature_names.extend([
                f'from_len_{base}',
                f'to_len_{base}',
                f'len_delta_{base}',
                f'len_abs_delta_{base}',
                f'same_{base}',
            ])

    return TransitionFeatureSpec(
        base_columns=base_columns,
        numeric_columns=numeric_columns,
        text_columns=text_columns,
        feature_names=feature_names,
    )


def _pair_feature_frame(df: pd.DataFrame, spec: TransitionFeatureSpec) -> pd.DataFrame:
    features: dict[str, pd.Series | np.ndarray] = {}

    for base in spec.numeric_columns:
        from_values = _numeric_series(df[f'from_{base}'], base).fillna(0.0).astype(float)
        to_values = _numeric_series(df[f'to_{base}'], base).fillna(0.0).astype(float)
        delta = to_values - from_values
        features[f'from_{base}'] = from_values
        features[f'to_{base}'] = to_values
        features[f'delta_{base}'] = delta
        features[f'abs_delta_{base}'] = delta.abs()
        features[f'mean_{base}'] = 0.5 * (from_values + to_values)

    for base in spec.text_columns:
        from_values = _text_series(df[f'from_{base}'])
        to_values = _text_series(df[f'to_{base}'])
        from_lengths = from_values.str.len().fillna(0).astype(float)
        to_lengths = to_values.str.len().fillna(0).astype(float)
        length_delta = to_lengths - from_lengths
        features[f'from_len_{base}'] = from_lengths
        features[f'to_len_{base}'] = to_lengths
        features[f'len_delta_{base}'] = length_delta
        features[f'len_abs_delta_{base}'] = length_delta.abs()
        features[f'same_{base}'] = (
            from_values.ne('') & to_values.ne('') & from_values.eq(to_values)
        ).astype(float)

    feature_frame = pd.DataFrame(features, index=df.index)
    return feature_frame.reindex(columns=spec.feature_names)


def _track_series(df: pd.DataFrame, base_column: str) -> pd.Series:
    candidates = _column_candidates(base_column)
    return _series_from_candidates(df, candidates)


def _track_numeric_array(df: pd.DataFrame, base_column: str) -> np.ndarray:
    series = _track_series(df, base_column)
    return _numeric_series(series, base_column).fillna(0.0).to_numpy(dtype=np.float32)


def _track_text_length_array(df: pd.DataFrame, base_column: str) -> np.ndarray:
    series = _track_series(df, base_column)
    return _text_series(series).str.len().fillna(0).to_numpy(dtype=np.float32)


def _track_feature_matrix(df: pd.DataFrame, spec: TransitionFeatureSpec) -> np.ndarray:
    rows = len(df)
    feature_matrix = np.zeros((rows * rows, len(spec.feature_names)), dtype=np.float32)
    column_index = 0

    for base in spec.numeric_columns:
        values = _track_numeric_array(df, base)
        left = np.repeat(values, rows)
        right = np.tile(values, rows)
        delta = right - left
        feature_matrix[:, column_index] = left
        feature_matrix[:, column_index + 1] = right
        feature_matrix[:, column_index + 2] = delta
        feature_matrix[:, column_index + 3] = np.abs(delta)
        feature_matrix[:, column_index + 4] = 0.5 * (left + right)
        column_index += 5

    for base in spec.text_columns:
        text_values = _text_series(_track_series(df, base)).to_numpy()
        lengths = _track_text_length_array(df, base)
        left = np.repeat(lengths, rows)
        right = np.tile(lengths, rows)
        delta = right - left
        same = np.repeat(text_values, rows) == np.tile(text_values, rows)
        same = np.asarray(same, dtype=np.float32)
        feature_matrix[:, column_index] = left
        feature_matrix[:, column_index + 1] = right
        feature_matrix[:, column_index + 2] = delta
        feature_matrix[:, column_index + 3] = np.abs(delta)
        feature_matrix[:, column_index + 4] = same
        column_index += 5

    return np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)


def _synthesize_negative_rows(
    positive_rows: pd.DataFrame,
    negative_count: int,
    random_state: int,
) -> pd.DataFrame:
    if negative_count <= 0:
        return pd.DataFrame(columns=positive_rows.columns)

    rng = np.random.default_rng(int(random_state))
    rows: list[dict[str, Any]] = []
    indices = np.arange(len(positive_rows))
    video_ids = positive_rows['video_id'].astype(str) if 'video_id' in positive_rows.columns else None

    for _ in range(int(negative_count)):
        left_index = int(rng.integers(0, len(positive_rows)))
        if video_ids is not None:
            different_video_indices = indices[video_ids.to_numpy() != video_ids.iloc[left_index]]
            candidate_indices = different_video_indices[different_video_indices != left_index]
        else:
            candidate_indices = indices[indices != left_index]
        if len(candidate_indices) == 0:
            candidate_indices = indices[indices != left_index]
        right_index = int(rng.choice(candidate_indices))
        left_row = positive_rows.iloc[left_index]
        right_row = positive_rows.iloc[right_index]
        synthetic_row = left_row.to_dict()
        for column in positive_rows.columns:
            if column.startswith('to_'):
                synthetic_row[column] = right_row[column]
        rows.append(synthetic_row)

    return pd.DataFrame(rows, columns=positive_rows.columns)


def _prepare_training_pairs(
    training_df: pd.DataFrame,
    negative_ratio: float,
    random_state: int,
) -> pd.DataFrame:
    if training_df.empty:
        raise ValueError('training data is empty')

    positive_rows = training_df.copy().reset_index(drop=True)
    positive_rows[MODEL_TARGET_COLUMN] = 1
    negative_count = int(round(len(positive_rows) * max(float(negative_ratio), 0.0)))
    negative_rows = _synthesize_negative_rows(positive_rows, negative_count, random_state)
    if not negative_rows.empty:
        negative_rows = negative_rows.assign(**{MODEL_TARGET_COLUMN: 0})

    combined = pd.concat([positive_rows, negative_rows], ignore_index=True, sort=False)
    if combined[MODEL_TARGET_COLUMN].nunique(dropna=False) < 2:
        raise ValueError('transition model training requires both positive and negative examples')
    return combined


def _normalize_device(device: str) -> str:
    normalized = str(device or '').strip().lower()
    if normalized in {'gpu', 'cuda'}:
        return 'cuda'
    if normalized in {'cpu'}:
        return 'cpu'
    if normalized in {'auto', ''}:
        return 'auto'
    raise ValueError(f'invalid transition model device: {device}')


def _import_torch_modules():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            'GPU transition model training requires PyTorch. Install torch with CUDA support and retry.'
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def _torch_cuda_available() -> bool:
    try:
        import torch
    except ModuleNotFoundError:
        return False
    return bool(torch.cuda.is_available())


def _build_torch_model(nn_module, input_dim: int, hidden_dims: tuple[int, int]):
    return nn_module.Sequential(
        nn_module.Linear(int(input_dim), int(hidden_dims[0])),
        nn_module.ReLU(),
        nn_module.Linear(int(hidden_dims[0]), int(hidden_dims[1])),
        nn_module.ReLU(),
        nn_module.Linear(int(hidden_dims[1]), 1),
    )


def _train_torch_transition_model(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    random_state: int,
    device: str,
) -> TorchTransitionModelPayload:
    torch, nn, DataLoader, TensorDataset = _import_torch_modules()
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA training was requested, but no CUDA-capable GPU is available to PyTorch.'
        )

    torch.manual_seed(int(random_state))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(random_state))

    model = _build_torch_model(nn, input_dim=features.shape[1], hidden_dims=_TORCH_HIDDEN_DIMS)
    torch_device = torch.device(device)
    model.to(torch_device)
    model.train()

    x_tensor = torch.as_tensor(features, dtype=torch.float32)
    y_tensor = torch.as_tensor(targets.reshape(-1, 1), dtype=torch.float32)
    dataset = TensorDataset(x_tensor, y_tensor)
    batch_size = min(_TORCH_BATCH_SIZE, max(len(dataset), 1))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    pos_count = max(float(targets.sum()), 1.0)
    neg_count = max(float(len(targets) - targets.sum()), 1.0)
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=torch_device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for _ in range(_TORCH_EPOCHS):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(torch_device, non_blocking=True)
            batch_y = batch_y.to(torch_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    state_dict = {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}
    return TorchTransitionModelPayload(
        input_dim=int(features.shape[1]),
        hidden_dims=_TORCH_HIDDEN_DIMS,
        state_dict=state_dict,
        trained_on_device=str(device),
    )


def _torch_predict_proba(
    payload: TorchTransitionModelPayload,
    features: np.ndarray,
    *,
    inference_device: str,
) -> np.ndarray:
    torch, nn, _, _ = _import_torch_modules()

    if features.ndim != 2:
        raise ValueError('expected a 2D feature matrix for torch inference')
    if int(features.shape[1]) != int(payload.input_dim):
        raise ValueError(
            f'torch transition model expects {payload.input_dim} features, got {features.shape[1]}'
        )
    if inference_device == 'cuda' and not torch.cuda.is_available():
        inference_device = 'cpu'

    model = _build_torch_model(nn, input_dim=payload.input_dim, hidden_dims=payload.hidden_dims)
    state_dict = {key: torch.as_tensor(value) for key, value in payload.state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(torch.device(inference_device))
    model.eval()

    with torch.no_grad():
        x_tensor = torch.as_tensor(features, dtype=torch.float32, device=torch.device(inference_device))
        logits = model(x_tensor).reshape(-1)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
    return probs.astype(np.float32)


def train_transition_model(
    training_df: pd.DataFrame,
    *,
    negative_ratio: float = DEFAULT_TRANSITION_MODEL_NEGATIVE_RATIO,
    random_state: int = DEFAULT_TRANSITION_MODEL_RANDOM_STATE,
    device: str = DEFAULT_TRANSITION_MODEL_DEVICE,
) -> TransitionModelArtifact:
    """Train a supervised transition model from positive rows plus synthetic negatives."""
    training_pairs = _prepare_training_pairs(training_df, negative_ratio=negative_ratio, random_state=random_state)
    feature_spec = _feature_spec_from_transition_rows(training_pairs)
    if not feature_spec.feature_names:
        raise ValueError('transition model training could not derive any features from the training data')

    features_df = _pair_feature_frame(training_pairs, feature_spec)
    features = np.nan_to_num(features_df.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = training_pairs[MODEL_TARGET_COLUMN].astype(int).to_numpy(dtype=np.float32)

    requested_device = _normalize_device(device)
    resolved_device = requested_device

    if requested_device == 'auto':
        resolved_device = 'cuda' if _torch_cuda_available() else 'cpu'

    if resolved_device == 'cuda':
        payload = _train_torch_transition_model(
            features,
            y,
            random_state=int(random_state),
            device='cuda',
        )
        backend = 'torch'
        model = payload
    else:
        backend = 'sklearn'
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            (
                'classifier',
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=2,
                    class_weight='balanced_subsample',
                    random_state=int(random_state),
                ),
            ),
        ])
        model.fit(features_df, y.astype(int))

    summary = {
        'positive_rows': int((y == 1).sum()),
        'negative_rows': int((y == 0).sum()),
        'feature_count': int(len(feature_spec.feature_names)),
        'base_columns': list(feature_spec.base_columns),
        'numeric_columns': list(feature_spec.numeric_columns),
        'text_columns': list(feature_spec.text_columns),
        'negative_ratio': float(negative_ratio),
        'backend': backend,
        'requested_device': requested_device,
        'resolved_device': resolved_device,
    }
    logger.info(
        'Trained transition model with %d positive and %d negative rows across %d features using %s (%s).',
        summary['positive_rows'],
        summary['negative_rows'],
        summary['feature_count'],
        summary['backend'],
        summary['resolved_device'],
    )
    return TransitionModelArtifact(
        pipeline=model,
        feature_spec=feature_spec,
        training_summary=summary,
        backend=backend,
        device=resolved_device,
    )


def save_transition_model(artifact: TransitionModelArtifact, path: str) -> None:
    out_path = os.fspath(path)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    joblib.dump(artifact, out_path)


def load_transition_model(path: str) -> TransitionModelArtifact:
    artifact = joblib.load(os.fspath(path))
    if not isinstance(artifact, TransitionModelArtifact):
        raise TypeError(f'unexpected transition model artifact type: {type(artifact)!r}')
    if not getattr(artifact, 'backend', ''):
        artifact.backend = 'sklearn'
    if not getattr(artifact, 'device', ''):
        artifact.device = 'cpu'
    return artifact


def load_transition_model_if_exists(path: str | None) -> TransitionModelArtifact | None:
    if not path:
        return None
    model_path = os.fspath(path)
    if not os.path.exists(model_path):
        return None
    return load_transition_model(model_path)


def score_transition_matrix(
    artifact: TransitionModelArtifact,
    df: pd.DataFrame,
) -> np.ndarray:
    if df.empty:
        return np.zeros((0, 0), dtype=np.float32)

    scored_df = add_sentiment_features(df)
    feature_matrix = _track_feature_matrix(scored_df, artifact.feature_spec)

    if getattr(artifact, 'backend', 'sklearn') == 'torch':
        inference_device = 'cuda' if str(getattr(artifact, 'device', 'cpu')) == 'cuda' and _torch_cuda_available() else 'cpu'
        probabilities = _torch_predict_proba(
            artifact.pipeline,
            feature_matrix,
            inference_device=inference_device,
        )
    else:
        feature_frame = pd.DataFrame(feature_matrix, columns=artifact.feature_spec.feature_names)
        probabilities = artifact.pipeline.predict_proba(feature_frame)[:, 1]

    score_matrix = probabilities.reshape(len(scored_df), len(scored_df)).astype(np.float32)
    np.fill_diagonal(score_matrix, 0.0)
    return np.clip(score_matrix, 0.0, 1.0)
