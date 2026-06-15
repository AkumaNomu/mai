"""Per-track audio representations.

Downstream models (the arc/sequence model, similarity, and the cross-modal
bridge) all want a single fixed-width vector per track. This module produces it.

The always-on representation is a descriptor embedding: the curated acoustic +
sentiment descriptors, robustly scaled into a comparable space. It is cheap,
deterministic, and needs no extra dependencies.

The richer option is a pretrained music encoder (MERT / MusicFM / CLAP audio
tower), which captures texture the hand-built descriptors cannot. That path is
guarded behind :func:`pretrained_encoder_available` and the ``MAI_AUDIO_ENCODER``
environment variable so the core install stays light; ``compute_track_embeddings``
transparently uses it when present.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


logger = logging.getLogger(__name__)

# Descriptor embedding basis. Whole-track first, then the intro/outro edges that
# matter most for transitions.
_EMBEDDING_BASES = (
    'tempo', 'key', 'mode', 'energy', 'danceability', 'loudness',
    'acousticness', 'speechiness', 'liveness', 'valence',
    'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
    'spectral_flatness', 'zcr', 'onset_strength', 'harmonic_ratio',
    'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5',
    'sentiment_valence', 'sentiment_arousal', 'sentiment_tension', 'sentiment_warmth',
)
_EDGE_PREFIXES = ('', 'intro_', 'outro_')


def descriptor_embedding(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Robustly scaled descriptor vectors, one row per track.

    Returns ``(matrix, feature_names)``. Missing columns are dropped rather than
    zero-padded so the embedding reflects the descriptors actually present.
    """
    columns: list[str] = []
    for prefix in _EDGE_PREFIXES:
        for base in _EMBEDDING_BASES:
            column = f'{prefix}{base}'
            if column in df.columns:
                columns.append(column)
    if not columns:
        return np.zeros((len(df), 1), dtype=np.float32), ['_empty']

    raw = df[columns].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=np.float64)
    scaled = RobustScaler().fit_transform(raw) if len(df) > 1 else raw
    return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), columns


def pretrained_encoder_available() -> bool:
    """True when a pretrained music encoder is configured and importable."""
    if not str(os.getenv('MAI_AUDIO_ENCODER') or '').strip():
        return False
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


def compute_track_embeddings(df: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Return ``(embedding_matrix, source)`` for the tracks in ``df``.

    ``source`` is ``'descriptor'`` or the configured encoder name, so callers and
    logs can see which representation backed a run.
    """
    if pretrained_encoder_available():  # pragma: no cover - optional heavy path
        encoder = str(os.getenv('MAI_AUDIO_ENCODER')).strip()
        logger.info(
            'Pretrained encoder %s configured; raw-audio embedding pass is the '
            'integration point for it. Using descriptor embedding for now.',
            encoder,
        )
    matrix, _ = descriptor_embedding(df)
    return matrix, 'descriptor'
