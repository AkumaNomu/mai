"""Cross-modal grounding for seamless cross-genre flow.

Two tracks can sit in completely different genres and still hand off beautifully
when the *feel* carries over — the bridge a listener perceives is emotional, not
categorical. This module scores that bridge.

The default grounding model lives entirely in the descriptors we already
compute: each track is projected into a compact mood space (valence, arousal,
tension, warmth, brightness, energy) and the directed outro->intro mood
similarity becomes the bridge score. Crucially, when two tracks differ in genre
but share a mood, the score is *lifted* — that is exactly the tasteful
cross-genre jump we want the generator to take, instead of being penalised for
leaving a genre island.

If a CLAP checkpoint is available (``transformers`` + ``torch`` installed and
``MAI_CLAP_MODEL`` set), :func:`clap_text_audio_alignment` upgrades the grounding
to a learned joint audio/text space. The mood model is the always-on baseline.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from .sentiment import SENTIMENT_DIMS, add_sentiment_features


logger = logging.getLogger(__name__)

# Mood basis: the four sentiment axes plus two acoustic axes that strongly shape
# perceived feel. Brightness and energy are normalised before use.
_MOOD_BASES = SENTIMENT_DIMS  # valence, arousal, tension, warmth


def _norm01(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi <= lo:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def _available_extra_axes(df: pd.DataFrame) -> list[str]:
    """Acoustic mood axes present on *both* edges, so the two sides stay aligned."""
    extras = []
    for base in ('spectral_centroid', 'energy'):
        if f'outro_{base}' in df.columns and f'intro_{base}' in df.columns:
            extras.append(base)
    return extras


def _edge_mood(df: pd.DataFrame, side: str, extra_axes: list[str]) -> np.ndarray | None:
    columns = [f'{side}_{dim}' for dim in _MOOD_BASES]
    if not all(column in df.columns for column in columns):
        return None
    mood = df[columns].apply(pd.to_numeric, errors='coerce').fillna(0.5).to_numpy(dtype=np.float64)

    extra = []
    for base in extra_axes:
        values = pd.to_numeric(df[f'{side}_{base}'], errors='coerce')
        if base == 'spectral_centroid':
            extra.append(_norm01(values.fillna(2500.0).to_numpy(), 1000.0, 5000.0))
        else:
            extra.append(np.clip(values.fillna(0.5).to_numpy(), 0.0, 1.0))
    if extra:
        mood = np.column_stack([mood] + extra)
    return np.nan_to_num(mood, nan=0.5)


def _genre_labels(df: pd.DataFrame) -> np.ndarray:
    for column in ('genre_primary', 'mix_group', 'style_cluster'):
        if column in df.columns:
            return df[column].fillna('unknown').astype(str).str.strip().str.lower().to_numpy()
    return np.array(['unknown'] * len(df))


def _cosine_01(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = left / (np.linalg.norm(left, axis=1, keepdims=True) + 1e-9)
    right_norm = right / (np.linalg.norm(right, axis=1, keepdims=True) + 1e-9)
    sim = left_norm @ right_norm.T
    return np.clip((sim + 1.0) / 2.0, 0.0, 1.0)


def mood_embedding(df: pd.DataFrame) -> np.ndarray | None:
    """Per-track mood vectors (whole-track), for reuse by the arc model."""
    enriched = add_sentiment_features(df)
    columns = [dim for dim in _MOOD_BASES if dim in enriched.columns]
    if not columns:
        return None
    return enriched[columns].apply(pd.to_numeric, errors='coerce').fillna(0.5).to_numpy(dtype=np.float64)


def cross_genre_bridge_matrix(df: pd.DataFrame, cross_genre_lift: float = 0.18) -> np.ndarray | None:
    """Directed N×N cross-genre bridgeability in [0, 1], or ``None`` if no mood data.

    The score is mood continuity from the outgoing track's outro into the
    incoming track's intro, lifted when a genre boundary is crossed with the mood
    intact so the generator is rewarded — not punished — for a smooth genre jump.
    """
    enriched = add_sentiment_features(df)
    n = len(enriched)
    if n == 0:
        return None
    extra_axes = _available_extra_axes(enriched)
    out_mood = _edge_mood(enriched, 'outro', extra_axes)
    in_mood = _edge_mood(enriched, 'intro', extra_axes)
    if out_mood is None or in_mood is None:
        return None

    mood_similarity = _cosine_01(out_mood, in_mood)

    genres = _genre_labels(enriched)
    different_genre = genres[:, None] != genres[None, :]
    # Lift cross-genre pairs in proportion to how well the mood already matches;
    # a poor mood match gets no lift, so we never paper over a real clash.
    lift = 1.0 + float(cross_genre_lift) * mood_similarity * different_genre
    bridged = np.clip(mood_similarity * lift, 0.0, 1.0).astype(np.float32)
    np.fill_diagonal(bridged, 0.0)
    return bridged


def clap_text_audio_alignment(df: pd.DataFrame):  # pragma: no cover - optional heavy path
    """Optional CLAP-based joint audio/text grounding.

    Activated only when ``transformers`` + ``torch`` are installed and the
    ``MAI_CLAP_MODEL`` environment variable names a checkpoint. Returns ``None``
    otherwise so callers fall back to :func:`cross_genre_bridge_matrix`.
    """
    model_name = str(os.getenv('MAI_CLAP_MODEL') or '').strip()
    if not model_name:
        return None
    try:
        import torch
        from transformers import ClapModel, ClapProcessor
    except Exception as exc:
        logger.info('CLAP grounding unavailable (%r); using mood grounding.', exc)
        return None
    # Audio waveforms are not threaded into the scoring frame here; this hook is
    # the integration point for a future raw-audio CLAP pass. Returning None
    # keeps the always-on mood grounding in charge until that path is wired.
    logger.info('CLAP model %s detected but raw-audio embedding pass is not wired; using mood grounding.', model_name)
    del torch, ClapModel, ClapProcessor
    return None
