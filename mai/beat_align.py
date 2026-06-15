"""Beat-grid and phrase-alignment scoring for the splice point.

A clean DJ handoff is beatmatched (tempos lock, octave-aware) and phrase-aware:
the incoming downbeat lands where the outgoing phrase resolves. We do not have
the absolute beat *phase* of each track here (that needs sample-accurate beat
tracking on the raw waveform — see ``audio_analysis`` for the hook that would
supply it), so this module scores beat-grid *compatibility* from the cached edge
descriptors: octave-aware tempo lock, mutual beat-grid stability, and
downbeat-to-downbeat strength across the cut. It is a directed outro -> intro
score in [0, 1].
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .musical_features import tempo_octave_closeness


def _edge_array(df: pd.DataFrame, side: str, base: str, default: float = 0.0) -> np.ndarray | None:
    column = f'{side}_{base}'
    if column in df.columns:
        values = pd.to_numeric(df[column], errors='coerce').fillna(default)
        return np.nan_to_num(values.to_numpy(dtype=np.float64), nan=default)
    return None


def phrase_alignment_matrix(df: pd.DataFrame) -> np.ndarray | None:
    """Directed N×N beat/phrase compatibility, or ``None`` if data is missing.

    Combines:
      * octave-aware tempo lock (you cannot phrase-match what you cannot beatmatch),
      * joint beat-grid stability (both sides must hold a steady grid to align),
      * downbeat handoff strength (strong outgoing resolve into strong incoming
        downbeat), softened when the outgoing tail leaves silence to drop into.
    """
    n = len(df)
    if n == 0:
        return None

    out_tempo = _edge_array(df, 'outro', 'tempo')
    in_tempo = _edge_array(df, 'intro', 'tempo')
    if out_tempo is None or in_tempo is None:
        return None

    out_beat = _edge_array(df, 'outro', 'beat_stability', default=0.5)
    in_beat = _edge_array(df, 'intro', 'beat_stability', default=0.5)
    out_down = _edge_array(df, 'outro', 'downbeat_strength', default=0.0)
    in_down = _edge_array(df, 'intro', 'downbeat_strength', default=0.0)
    out_tail = _edge_array(df, 'outro', 'tail_silence_s', default=0.0)
    if any(value is None for value in (out_beat, in_beat, out_down, in_down)):
        return None

    from_tempo = np.repeat(out_tempo, n)
    to_tempo = np.tile(in_tempo, n)
    tempo_lock, _ = tempo_octave_closeness(from_tempo, to_tempo)
    tempo_lock = tempo_lock.reshape(n, n)

    grid_stability = np.outer(np.clip(out_beat, 0.0, 1.0), np.clip(in_beat, 0.0, 1.0))
    downbeat_handoff = np.outer(np.clip(out_down, 0.0, 1.0), np.clip(in_down, 0.0, 1.0))

    # When the outgoing track tails into silence, a strong incoming downbeat can
    # land cleanly even without a downbeat-to-downbeat butt-cut, so relax the
    # downbeat requirement proportionally to available tail silence.
    tail_norm = np.clip(out_tail / 4.0, 0.0, 1.0)  # ~4s of tail fully relaxes it
    downbeat_relaxed = downbeat_handoff + (1.0 - downbeat_handoff) * tail_norm[:, None]

    score = (
        0.45 * tempo_lock
        + 0.30 * grid_stability
        + 0.25 * downbeat_relaxed
    )
    score = np.clip(score, 0.0, 1.0).astype(np.float32)
    np.fill_diagonal(score, 0.0)
    return score
