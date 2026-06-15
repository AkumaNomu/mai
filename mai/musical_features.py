"""Domain-aware pairwise transition features.

These functions encode the parts of a song-to-song handoff that a DJ actually
reasons about: harmonic compatibility on the Camelot wheel, octave-aware tempo
matching, and the energy / mood trajectory across the cut. They are shared by
both the transition scorer (``playlist_generation``) and the supervised model
(``transition_model``) so that the features a model trains on are exactly the
features it is scored with.

Every primitive operates on two equally shaped arrays ``from_values`` and
``to_values`` (the outgoing side and the incoming side of a transition) and
returns an array of the same shape. The row-wise training path passes the
``from_*`` / ``to_*`` columns of the transition table; the matrix scoring path
passes ``np.repeat`` / ``np.tile`` expansions of a per-track column. Because the
math is identical in both cases, train/serve skew is impossible by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Pitch classes used for harmonic-mixing math. The Camelot wheel is the circle
# of fifths indexed 1..12 with a major (B) and minor (A) ring; two tracks mix
# harmonically when they share a Camelot code or are adjacent on the wheel /
# relative major-minor pairs.
_FIFTHS_ORDER = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]  # C, G, D, A, ... by fifths
_PITCH_TO_FIFTH = {pitch: index for index, pitch in enumerate(_FIFTHS_ORDER)}

# Base scalar columns the engineered features read from. Both training rows and
# per-track frames are expected to expose these (missing columns degrade to 0).
MUSICAL_INPUT_BASES = (
    'key',
    'mode',
    'tempo',
    'energy',
    'loudness',
    'danceability',
    'onset_strength',
    'sentiment_arousal',
    'sentiment_valence',
    'sentiment_tension',
)

# Stable, ordered list of engineered feature names. Both code paths emit these
# columns in this order so the model's feature vector never shifts.
MUSICAL_FEATURE_NAMES = (
    'camelot_compatibility',
    'circle_of_fifths_distance',
    'relative_key_match',
    'tempo_octave_closeness',
    'tempo_abs_log_ratio',
    'energy_rise',
    'energy_abs_jump',
    'arousal_rise',
    'arousal_abs_jump',
    'valence_shift_abs',
    'tension_release',
    'groove_continuity',
)


def _as_float_array(values: np.ndarray | pd.Series) -> np.ndarray:
    array = np.asarray(pd.to_numeric(pd.Series(values), errors='coerce'), dtype=np.float64)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def _circle_distance(from_key: np.ndarray, to_key: np.ndarray) -> np.ndarray:
    """Shortest hop count between two pitch classes on the circle of fifths (0..6)."""
    from_pitch = np.mod(np.rint(from_key).astype(int), 12)
    to_pitch = np.mod(np.rint(to_key).astype(int), 12)
    from_fifth = np.vectorize(_PITCH_TO_FIFTH.get)(from_pitch)
    to_fifth = np.vectorize(_PITCH_TO_FIFTH.get)(to_pitch)
    raw = np.abs(from_fifth - to_fifth)
    return np.minimum(raw, 12 - raw).astype(np.float64)


def camelot_compatibility(
    from_key: np.ndarray,
    from_mode: np.ndarray,
    to_key: np.ndarray,
    to_mode: np.ndarray,
) -> np.ndarray:
    """Harmonic-mix score in [0, 1] following standard Camelot adjacency rules.

    1.00 same key+mode, 0.90 relative major/minor, 0.80 one step on the wheel,
    decaying for larger harmonic distances. This is the single most important
    cue for "does this blend sound in-key".
    """
    from_key = np.mod(np.rint(_as_float_array(from_key)).astype(int), 12)
    to_key = np.mod(np.rint(_as_float_array(to_key)).astype(int), 12)
    from_major = np.rint(_as_float_array(from_mode)).astype(int) == 1
    to_major = np.rint(_as_float_array(to_mode)).astype(int) == 1

    distance = _circle_distance(from_key, to_key)
    same_mode = from_major == to_major
    same_key = from_key == to_key

    # Relative major/minor share a key signature: A minor (pitch 9, minor) <-> C
    # major (pitch 0, major) sit 3 semitones apart with opposite mode.
    semitone_gap = np.mod(to_key - from_key, 12)
    relative_pair = (~same_mode) & (
        (from_major & (semitone_gap == 9)) | ((~from_major) & (semitone_gap == 3))
    )

    score = np.full(from_key.shape, 0.0, dtype=np.float64)
    score = np.where(same_mode, np.clip(1.0 - distance / 6.0, 0.0, 1.0) * 0.85 + 0.15 * (distance <= 1), score)
    score = np.where(same_mode & same_key, 1.0, score)
    score = np.where(same_mode & (distance == 1), np.maximum(score, 0.80), score)
    score = np.where(relative_pair, np.maximum(score, 0.90), score)
    # Cross-mode but harmonically near still earns partial credit.
    score = np.where((~same_mode) & (~relative_pair), np.clip(0.65 - distance / 8.0, 0.0, 0.65), score)
    return np.clip(score, 0.0, 1.0)


def _relative_key_match(from_key, from_mode, to_key, to_mode) -> np.ndarray:
    from_key = np.mod(np.rint(_as_float_array(from_key)).astype(int), 12)
    to_key = np.mod(np.rint(_as_float_array(to_key)).astype(int), 12)
    from_major = np.rint(_as_float_array(from_mode)).astype(int) == 1
    semitone_gap = np.mod(to_key - from_key, 12)
    same = (from_major == (np.rint(_as_float_array(to_mode)).astype(int) == 1)) & (from_key == to_key)
    relative = (from_major & (semitone_gap == 9)) | ((~from_major) & (semitone_gap == 3))
    return (same | relative).astype(np.float64)


def tempo_octave_closeness(from_tempo: np.ndarray, to_tempo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Octave-aware tempo agreement.

    Returns ``(closeness, abs_log_ratio)``. A 70 BPM track mixes cleanly into a
    140 BPM track (double time), so the ratio is folded into ``[1/sqrt2, sqrt2]``
    before measuring distance. ``closeness`` is in [0, 1]; ``abs_log_ratio`` is
    the raw folded |log2 ratio| for the model to use directly.
    """
    from_tempo = _as_float_array(from_tempo)
    to_tempo = _as_float_array(to_tempo)
    safe_from = np.where(from_tempo > 1e-3, from_tempo, np.nan)
    safe_to = np.where(to_tempo > 1e-3, to_tempo, np.nan)
    ratio = safe_to / safe_from
    log_ratio = np.log2(ratio)
    folded = log_ratio - np.round(log_ratio)  # fold octaves -> [-0.5, 0.5]
    abs_log = np.abs(folded)
    closeness = np.exp(-abs_log / 0.08)  # ~half-credit at ~5.5% tempo drift
    closeness = np.where(np.isfinite(closeness), closeness, 0.0)
    abs_log = np.where(np.isfinite(abs_log), abs_log, 0.5)
    return np.clip(closeness, 0.0, 1.0), abs_log


def _directional_rise(from_values: np.ndarray, to_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from_values = _as_float_array(from_values)
    to_values = _as_float_array(to_values)
    delta = to_values - from_values
    return delta, np.abs(delta)


def musical_interaction_features(
    from_values: dict[str, np.ndarray],
    to_values: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Build the engineered transition features from outgoing/incoming arrays.

    ``from_values`` / ``to_values`` map each base in :data:`MUSICAL_INPUT_BASES`
    to an array. The two sides must share a shape. Returns a dict keyed by
    :data:`MUSICAL_FEATURE_NAMES`.
    """
    def col(side: dict[str, np.ndarray], base: str) -> np.ndarray:
        return _as_float_array(side.get(base, 0.0))

    camelot = camelot_compatibility(
        col(from_values, 'key'), col(from_values, 'mode'),
        col(to_values, 'key'), col(to_values, 'mode'),
    )
    circle = _circle_distance(
        np.rint(col(from_values, 'key')).astype(int),
        np.rint(col(to_values, 'key')).astype(int),
    )
    relative = _relative_key_match(
        col(from_values, 'key'), col(from_values, 'mode'),
        col(to_values, 'key'), col(to_values, 'mode'),
    )
    tempo_close, tempo_abs_log = tempo_octave_closeness(
        col(from_values, 'tempo'), col(to_values, 'tempo')
    )
    energy_rise, energy_abs = _directional_rise(col(from_values, 'energy'), col(to_values, 'energy'))
    arousal_rise, arousal_abs = _directional_rise(
        col(from_values, 'sentiment_arousal'), col(to_values, 'sentiment_arousal')
    )
    _, valence_abs = _directional_rise(
        col(from_values, 'sentiment_valence'), col(to_values, 'sentiment_valence')
    )
    # Releasing tension across the cut (high outgoing tension -> lower incoming)
    # is a hallmark of a satisfying handoff.
    tension_release = np.clip(
        col(from_values, 'sentiment_tension') - col(to_values, 'sentiment_tension') + 0.5, 0.0, 1.0
    )
    groove_continuity = np.exp(
        -np.abs(col(from_values, 'danceability') - col(to_values, 'danceability')) / 0.25
    )

    return {
        'camelot_compatibility': np.clip(camelot, 0.0, 1.0),
        'circle_of_fifths_distance': circle,
        'relative_key_match': relative,
        'tempo_octave_closeness': np.clip(tempo_close, 0.0, 1.0),
        'tempo_abs_log_ratio': tempo_abs_log,
        'energy_rise': energy_rise,
        'energy_abs_jump': energy_abs,
        'arousal_rise': arousal_rise,
        'arousal_abs_jump': arousal_abs,
        'valence_shift_abs': valence_abs,
        'tension_release': tension_release,
        'groove_continuity': np.clip(groove_continuity, 0.0, 1.0),
    }


def _track_base_array(df: pd.DataFrame, base: str) -> np.ndarray:
    """Per-track scalar for a base, preferring whole-track then intro/outro."""
    for column in (base, f'intro_{base}', f'outro_{base}'):
        if column in df.columns:
            return _as_float_array(df[column])
    return np.zeros(len(df), dtype=np.float64)


def harmonic_tempo_matrix(df: pd.DataFrame) -> np.ndarray | None:
    """Standalone directed N×N compatibility blending Camelot + tempo + groove.

    Uses the outgoing track's outro key/tempo into the incoming track's intro.
    Returns ``None`` when the required key/tempo columns are absent.
    """
    n = len(df)
    if n == 0:
        return None

    def edge(side: str, base: str) -> np.ndarray | None:
        column = f'{side}_{base}'
        if column in df.columns:
            return _as_float_array(df[column])
        if base in df.columns:
            return _as_float_array(df[base])
        return None

    out_key, in_key = edge('outro', 'key'), edge('intro', 'key')
    out_mode, in_mode = edge('outro', 'mode'), edge('intro', 'mode')
    out_tempo, in_tempo = edge('outro', 'tempo'), edge('intro', 'tempo')
    if out_key is None or in_key is None or out_tempo is None or in_tempo is None:
        return None
    if out_mode is None:
        out_mode = np.ones(n)
    if in_mode is None:
        in_mode = np.ones(n)

    from_key = np.repeat(out_key, n)
    to_key = np.tile(in_key, n)
    from_mode = np.repeat(out_mode, n)
    to_mode = np.tile(in_mode, n)
    from_tempo = np.repeat(out_tempo, n)
    to_tempo = np.tile(in_tempo, n)

    camelot = camelot_compatibility(from_key, from_mode, to_key, to_mode)
    tempo_close, _ = tempo_octave_closeness(from_tempo, to_tempo)
    blended = 0.6 * camelot + 0.4 * tempo_close
    matrix = blended.reshape(n, n).astype(np.float32)
    np.fill_diagonal(matrix, 0.0)
    return np.clip(matrix, 0.0, 1.0)
