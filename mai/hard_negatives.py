"""Hard-negative mining for the transition model.

The scraper only produces *positive* transitions (real DJ-chosen handoffs). The
old model paired every positive with a uniformly random cross-mix pair as its
negative. Random negatives are almost always a different genre/tempo/key, so the
classifier learns the trivial shortcut "are these two songs similar at all?"
instead of the thing we care about: "given two plausible neighbours, is *this*
the good handoff?".

Hard negatives fix that. For each positive A -> B we keep the outgoing track A
and swap in a replacement C that is *close to B in musical feature space* but was
never the curated next track and comes from a different source mix. The model is
then forced onto the decision boundary that actually matters. A small fraction of
easy random negatives is retained so the probabilities stay calibrated.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler


logger = logging.getLogger(__name__)

MODEL_TARGET_COLUMN = '__transition_target__'

# Numeric "to"-side descriptors used to measure how interchangeable two
# candidate next-tracks are. Kept compact and genre-agnostic on purpose.
_SIMILARITY_BASES = (
    'tempo',
    'key',
    'energy',
    'danceability',
    'loudness',
    'acousticness',
    'spectral_centroid',
    'onset_strength',
    'sentiment_valence',
    'sentiment_arousal',
    'sentiment_tension',
    'mfcc1',
    'mfcc2',
    'mfcc3',
)


def _first_present(df: pd.DataFrame, columns) -> str | None:
    for column in columns:
        if column in df.columns:
            return column
    return None


def _track_id_series(df: pd.DataFrame, side: str) -> pd.Series:
    column = _first_present(df, [f'{side}_video_id', f'{side}_resolved_video_id'])
    if column is None:
        # Fall back to a content hash of the side's title/artist so dedup still works.
        title = df.get(f'{side}_resolved_title', pd.Series('', index=df.index)).fillna('').astype(str)
        artist = df.get(f'{side}_resolved_artist', pd.Series('', index=df.index)).fillna('').astype(str)
        return (artist + '␟' + title).str.strip().str.lower()
    return df[column].fillna('').astype(str).str.strip()


def _to_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    columns = [f'to_{base}' for base in _SIMILARITY_BASES if f'to_{base}' in df.columns]
    if not columns:
        return np.zeros((len(df), 1), dtype=np.float64)
    matrix = df[columns].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=np.float64)
    matrix = RobustScaler().fit_transform(matrix)
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def _neighbour_index(features: np.ndarray, neighbours: int) -> np.ndarray:
    """Return an (n, k) index array of nearest rows (self included at column 0)."""
    n = features.shape[0]
    k = int(min(max(neighbours, 1) + 1, n))
    model = NearestNeighbors(n_neighbors=k, algorithm='auto')
    model.fit(features)
    _, indices = model.kneighbors(features)
    return indices


def mine_hard_negatives(
    positive_df: pd.DataFrame,
    *,
    negative_ratio: float = 1.0,
    hard_fraction: float = 0.8,
    neighbours: int = 12,
    random_state: int = 42,
    mix_id_column: str = 'video_id',
) -> pd.DataFrame:
    """Build a negatives table aligned to ``positive_df``'s columns.

    Each negative keeps a real outgoing (``from_*``) context and substitutes the
    incoming (``to_*``) track with a close-but-unchosen alternative.
    """
    if positive_df.empty:
        return pd.DataFrame(columns=positive_df.columns)

    rng = np.random.default_rng(int(random_state))
    positives = positive_df.reset_index(drop=True)
    n = len(positives)
    target_count = int(round(n * max(float(negative_ratio), 0.0)))
    if target_count <= 0:
        return pd.DataFrame(columns=positives.columns)

    mix_ids = (
        positives[mix_id_column].fillna('').astype(str).to_numpy()
        if mix_id_column in positives.columns else np.array([''] * n)
    )
    to_ids = _track_id_series(positives, 'to').to_numpy()
    from_ids = _track_id_series(positives, 'from').to_numpy()
    to_columns = [column for column in positives.columns if column.startswith('to_')]

    features = _to_feature_matrix(positives)
    neighbour_idx = _neighbour_index(features, neighbours)

    hard_target = int(round(target_count * float(np.clip(hard_fraction, 0.0, 1.0))))
    rows: list[dict] = []
    indices = np.arange(n)

    # --- hard negatives: nearest unchosen alternative from another mix ---
    left_order = rng.permutation(n)
    cursor = 0
    while len(rows) < hard_target and cursor < hard_target * 4:
        left = int(left_order[cursor % n])
        cursor += 1
        candidates = [
            idx for idx in neighbour_idx[left][1:]
            if mix_ids[idx] != mix_ids[left] and to_ids[idx] not in {to_ids[left], from_ids[left], ''}
        ]
        if not candidates:
            continue
        # Prefer the closest plausible alternative but keep some spread.
        choice = int(candidates[int(rng.integers(0, min(len(candidates), neighbours)))]) if candidates else None
        if choice is None:
            continue
        synthetic = positives.iloc[left].to_dict()
        donor = positives.iloc[choice]
        for column in to_columns:
            synthetic[column] = donor[column]
        rows.append(synthetic)

    hard_built = len(rows)

    # --- easy negatives: uniform random cross-mix pairs for calibration ---
    while len(rows) < target_count:
        left = int(rng.integers(0, n))
        pool = indices[(mix_ids != mix_ids[left]) & (to_ids != to_ids[left]) & (to_ids != '')]
        if pool.size == 0:
            pool = indices[indices != left]
        if pool.size == 0:
            break
        right = int(rng.choice(pool))
        synthetic = positives.iloc[left].to_dict()
        donor = positives.iloc[right]
        for column in to_columns:
            synthetic[column] = donor[column]
        rows.append(synthetic)

    logger.info(
        'Hard-negative mining: %d negatives (%d hard, %d easy) from %d positives.',
        len(rows), hard_built, len(rows) - hard_built, n,
    )
    negatives = pd.DataFrame(rows, columns=positives.columns)
    return negatives


def build_training_table(
    positive_df: pd.DataFrame,
    *,
    negative_ratio: float = 1.0,
    hard_fraction: float = 0.8,
    neighbours: int = 12,
    random_state: int = 42,
    mix_id_column: str = 'video_id',
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return ``(labelled_df, stats)`` ready for model training."""
    if positive_df.empty:
        raise ValueError('training data is empty')

    positives = positive_df.copy().reset_index(drop=True)
    positives[MODEL_TARGET_COLUMN] = 1
    negatives = mine_hard_negatives(
        positives,
        negative_ratio=negative_ratio,
        hard_fraction=hard_fraction,
        neighbours=neighbours,
        random_state=random_state,
        mix_id_column=mix_id_column,
    )
    if not negatives.empty:
        negatives[MODEL_TARGET_COLUMN] = 0

    combined = pd.concat([positives, negatives], ignore_index=True, sort=False)
    if combined[MODEL_TARGET_COLUMN].nunique(dropna=False) < 2:
        raise ValueError('transition model training requires both positive and negative examples')

    stats = {
        'positive_rows': int((combined[MODEL_TARGET_COLUMN] == 1).sum()),
        'negative_rows': int((combined[MODEL_TARGET_COLUMN] == 0).sum()),
    }
    return combined, stats
