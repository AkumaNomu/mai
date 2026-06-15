"""Set-level arc modelling: the *flow* of a whole playlist, not just pairs.

Pairwise transition scores answer "do these two songs touch well?". They cannot
answer "why this song *now*?" — the global shape of a great set, which builds,
peaks, and cools. This module scores that shape.

The always-on model is an energy/arousal *arc fit*: a playlist's mood curve is
compared against a target arc (rise -> peak -> cool-down by default). It needs no
training and is used to refine ordering after the transition-driven search.

The learned model is an optional sequence classifier (a small GRU over track
embeddings) trained on real DJ mix orderings versus shuffled ones, so it learns
what a *real* progression looks like. It activates only when PyTorch is present;
otherwise the arc fit is authoritative.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .embeddings import compute_track_embeddings
from .sentiment import add_sentiment_features


logger = logging.getLogger(__name__)

DEFAULT_ARC_MODEL_PATH = os.path.join('data', 'cache', 'arc_model.joblib')

# Named target arcs over normalised position p in [0, 1].
_ARC_TARGETS = {
    'rise_peak_cool': lambda p: np.clip(0.40 + 0.55 * np.sin(np.pi * np.power(p, 0.85)), 0.0, 1.0),
    'rise': lambda p: np.clip(0.30 + 0.65 * p, 0.0, 1.0),
    'plateau': lambda p: np.full_like(p, 0.65),
    'cool_down': lambda p: np.clip(0.85 - 0.55 * p, 0.0, 1.0),
}
DEFAULT_ARC = 'rise_peak_cool'


def _arousal_curve(df: pd.DataFrame, path: list[int]) -> np.ndarray:
    enriched = add_sentiment_features(df)
    if 'sentiment_arousal' in enriched.columns:
        values = pd.to_numeric(enriched['sentiment_arousal'], errors='coerce').fillna(0.5).to_numpy()
    elif 'energy' in enriched.columns:
        values = pd.to_numeric(enriched['energy'], errors='coerce').fillna(0.5).to_numpy()
    else:
        values = np.full(len(enriched), 0.5)
    return np.clip(values[path], 0.0, 1.0)


def arc_fit(df: pd.DataFrame, path: list[int], arc: str = DEFAULT_ARC) -> float:
    """How well a playlist's mood curve matches the target arc, in [0, 1]."""
    if len(path) < 3:
        return 1.0
    curve = _arousal_curve(df, path)
    positions = np.linspace(0.0, 1.0, num=len(curve))
    target = _ARC_TARGETS.get(arc, _ARC_TARGETS[DEFAULT_ARC])(positions)
    shape_error = float(np.mean(np.abs(curve - target)))
    # Penalise jagged, non-monotone-within-segment hops on top of shape error.
    roughness = float(np.mean(np.abs(np.diff(curve)))) if len(curve) > 1 else 0.0
    return float(np.clip(1.0 - (0.8 * shape_error + 0.2 * roughness), 0.0, 1.0))


@dataclass(slots=True)
class ArcModelArtifact:
    backend: str = 'heuristic'
    arc: str = DEFAULT_ARC
    input_dim: int = 0
    hidden_dim: int = 64
    state_dict: dict[str, np.ndarray] = field(default_factory=dict)
    training_summary: dict[str, Any] = field(default_factory=dict)

    def score_path(self, df: pd.DataFrame, path: list[int]) -> float:
        return score_ordering(self, df, path)


def heuristic_arc_model(arc: str = DEFAULT_ARC) -> ArcModelArtifact:
    return ArcModelArtifact(backend='heuristic', arc=arc)


def score_ordering(artifact: ArcModelArtifact | None, df: pd.DataFrame, path: list[int]) -> float:
    """Score a candidate ordering in [0, 1]. Falls back to arc fit when needed."""
    arc = artifact.arc if artifact is not None else DEFAULT_ARC
    if artifact is None or artifact.backend != 'torch' or not artifact.state_dict:
        return arc_fit(df, path, arc=arc)
    try:  # pragma: no cover - optional heavy path
        embeddings, _ = compute_track_embeddings(df)
        prob = _torch_score_sequence(artifact, embeddings[path])
        # Blend learned progression score with arc fit so both shape and learned
        # structure count; arc fit is the stable prior.
        return float(np.clip(0.6 * prob + 0.4 * arc_fit(df, path, arc=arc), 0.0, 1.0))
    except Exception:
        logger.exception('Arc model torch scoring failed; using arc fit.')
        return arc_fit(df, path, arc=arc)


# --------------------------------------------------------------------------- #
# Optional learned model (PyTorch). Off the default path; trained by mai.train. #
# --------------------------------------------------------------------------- #

def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def _reconstruct_mix_sequences(training_df: pd.DataFrame) -> list[list[int]]:
    """Group transition rows into per-mix ordered row-index sequences."""
    if 'video_id' not in training_df.columns:
        return []
    order_column = 'from_position' if 'from_position' in training_df.columns else None
    sequences: list[list[int]] = []
    for _, group in training_df.groupby('video_id', sort=False):
        ordered = group.sort_values(order_column, kind='stable') if order_column else group
        indices = ordered.index.tolist()
        if len(indices) >= 3:
            sequences.append(indices)
    return sequences


def train_arc_model(
    training_df: pd.DataFrame,
    *,
    arc: str = DEFAULT_ARC,
    random_state: int = 42,
    epochs: int = 40,
) -> ArcModelArtifact:
    """Train the sequence arc model; returns a heuristic artifact if torch absent."""
    if not _torch_available():
        logger.info('PyTorch not available; arc model uses the heuristic arc fit.')
        return heuristic_arc_model(arc)

    sequences = _reconstruct_mix_sequences(training_df.reset_index(drop=True))
    if len(sequences) < 4:
        logger.info('Too few reconstructable mixes (%d) for a learned arc model; using heuristic.', len(sequences))
        return heuristic_arc_model(arc)

    import torch
    from torch import nn

    frame = add_sentiment_features(training_df.reset_index(drop=True))
    embeddings, _ = compute_track_embeddings(frame)
    input_dim = int(embeddings.shape[1])
    rng = np.random.default_rng(int(random_state))

    positives = [embeddings[seq] for seq in sequences]
    negatives = []
    for seq in sequences:
        shuffled = list(seq)
        rng.shuffle(shuffled)
        negatives.append(embeddings[shuffled])

    samples = [(seq, 1.0) for seq in positives] + [(seq, 0.0) for seq in negatives]

    class _ArcGRU(nn.Module):
        def __init__(self, dim: int, hidden: int):
            super().__init__()
            self.gru = nn.GRU(dim, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))

        def forward(self, x):
            output, _ = self.gru(x)
            return self.head(output.mean(dim=1))

    hidden_dim = 64
    torch.manual_seed(int(random_state))
    model = _ArcGRU(input_dim, hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(int(epochs)):
        order = rng.permutation(len(samples))
        for index in order:
            sequence, label = samples[index]
            x = torch.as_tensor(sequence[None, :, :], dtype=torch.float32)
            y = torch.as_tensor([[label]], dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    state_dict = {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}
    summary = {'backend': 'torch', 'mixes': len(sequences), 'input_dim': input_dim, 'epochs': int(epochs)}
    logger.info('Trained learned arc model on %d mixes (input_dim=%d).', len(sequences), input_dim)
    return ArcModelArtifact(
        backend='torch', arc=arc, input_dim=input_dim, hidden_dim=hidden_dim,
        state_dict=state_dict, training_summary=summary,
    )


def _torch_score_sequence(artifact: ArcModelArtifact, sequence: np.ndarray) -> float:  # pragma: no cover
    import torch
    from torch import nn

    class _ArcGRU(nn.Module):
        def __init__(self, dim: int, hidden: int):
            super().__init__()
            self.gru = nn.GRU(dim, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))

        def forward(self, x):
            output, _ = self.gru(x)
            return self.head(output.mean(dim=1))

    model = _ArcGRU(artifact.input_dim, artifact.hidden_dim)
    model.load_state_dict({key: torch.as_tensor(value) for key, value in artifact.state_dict.items()})
    model.eval()
    if sequence.shape[1] != artifact.input_dim:
        raise ValueError(f'arc model expects input_dim={artifact.input_dim}, got {sequence.shape[1]}')
    with torch.no_grad():
        logits = model(torch.as_tensor(sequence[None, :, :], dtype=torch.float32))
        return float(torch.sigmoid(logits).item())


def save_arc_model(artifact: ArcModelArtifact, path: str) -> None:
    directory = os.path.dirname(os.fspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    joblib.dump(artifact, os.fspath(path))


def load_arc_model_if_exists(path: str | None) -> ArcModelArtifact | None:
    if not path or not os.path.exists(os.fspath(path)):
        return None
    artifact = joblib.load(os.fspath(path))
    return artifact if isinstance(artifact, ArcModelArtifact) else None
