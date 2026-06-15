"""Set-level arc modelling: the *flow* of a whole playlist, not just pairs.

Pairwise transition scores answer "do these two songs touch well?". They cannot
answer "why this song *now*?" — the global shape of a great set, which builds,
peaks, and cools. This module scores that shape.

The always-on model is an energy/arousal *arc fit*: a playlist's mood curve is
compared against a target arc (rise -> peak -> cool-down by default). It needs no
training and is used to refine ordering after the transition-driven search.

The learned model has two tiers, both trained on real DJ mix orderings versus
shuffled ones so they learn what a *real* progression looks like:

* A torch-free **order classifier** (logistic regression over a handful of
  order-summary features). It is the always-on learned tier and is deliberately
  low-data: a few scraped mixes, each contrasted against many shuffles, are
  enough because the feature space is tiny. This is the practical answer to
  "learn from playlist *order*, not labelled pairs" when data is scarce.
* An optional **GRU** over track embeddings, used only when PyTorch is present
  *and* there are enough reconstructable mixes to justify it.

When neither can be trained the arc fit is authoritative.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .embeddings import compute_track_embeddings
from .sentiment import add_sentiment_features


logger = logging.getLogger(__name__)

DEFAULT_ARC_MODEL_PATH = os.path.join('data', 'cache', 'arc_model.joblib')

# Torch-free order classifier: minimum mixes to attempt it, and how many shuffled
# negatives to contrast each real ordering against. Few mixes are fine because the
# order-summary feature space is tiny.
_ORDER_MIN_MIXES = 2
_ORDER_SHUFFLES_PER_MIX = 8

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
    # Torch-free order classifier (sklearn pipeline) and its feature width.
    order_model: Any = None
    feature_dim: int = 0

    def score_path(self, df: pd.DataFrame, path: list[int]) -> float:
        return score_ordering(self, df, path)


def heuristic_arc_model(arc: str = DEFAULT_ARC) -> ArcModelArtifact:
    return ArcModelArtifact(backend='heuristic', arc=arc)


def _blend_learned(prob: float, df: pd.DataFrame, path: list[int], arc: str) -> float:
    # Blend learned progression score with arc fit so both the learned structure
    # and the target shape count; arc fit is the stable prior.
    return float(np.clip(0.6 * float(prob) + 0.4 * arc_fit(df, path, arc=arc), 0.0, 1.0))


def score_ordering(artifact: ArcModelArtifact | None, df: pd.DataFrame, path: list[int]) -> float:
    """Score a candidate ordering in [0, 1]. Falls back to arc fit when needed."""
    arc = artifact.arc if artifact is not None else DEFAULT_ARC
    if artifact is None or len(path) < 3:
        return arc_fit(df, path, arc=arc)

    backend = getattr(artifact, 'backend', 'heuristic')

    if backend == 'order_clf' and getattr(artifact, 'order_model', None) is not None:
        try:
            embeddings, _ = compute_track_embeddings(df)
            features = _order_summary_features(df, path, embeddings).reshape(1, -1)
            if features.shape[1] != int(getattr(artifact, 'feature_dim', features.shape[1])):
                return arc_fit(df, path, arc=arc)
            prob = float(artifact.order_model.predict_proba(features)[0, 1])
            return _blend_learned(prob, df, path, arc)
        except Exception:
            logger.exception('Order arc model scoring failed; using arc fit.')
            return arc_fit(df, path, arc=arc)

    if backend != 'torch' or not artifact.state_dict:
        return arc_fit(df, path, arc=arc)
    try:  # pragma: no cover - optional heavy path
        embeddings, _ = compute_track_embeddings(df)
        prob = _torch_score_sequence(artifact, embeddings[path])
        return _blend_learned(prob, df, path, arc)
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


def _order_summary_features(
    df: pd.DataFrame,
    path: list[int],
    embeddings: np.ndarray | None = None,
) -> np.ndarray:
    """A small, fixed-width description of an ordering's *shape* and smoothness.

    Deliberately low-dimensional (11 scalars) so the order classifier generalises
    from a handful of mixes instead of overfitting per-track embeddings.
    """
    curve = _arousal_curve(df, path)
    n = len(curve)
    if n == 0:
        curve = np.array([0.5], dtype=np.float64)
        n = 1
    diffs = np.diff(curve) if n > 1 else np.zeros(1, dtype=np.float64)
    positions = np.linspace(0.0, 1.0, num=n)
    target = _ARC_TARGETS[DEFAULT_ARC](positions)

    feats = [
        float(np.mean(np.abs(diffs))),                       # roughness
        float(np.std(curve)),                                # dynamic spread
        float(curve[0]),                                     # opener energy
        float(curve[-1]),                                    # closer energy
        float(np.max(curve) - np.min(curve)),                # energy span
        float(np.argmax(curve)) / float(max(n - 1, 1)),      # peak position
        float(np.mean((diffs > 0).astype(np.float64))) if diffs.size else 0.5,  # rise share
        float(np.mean(np.abs(curve - target))),              # arc shape error
    ]

    if embeddings is not None and len(path) > 1:
        seq = embeddings[path]
        numerator = np.sum(seq[:-1] * seq[1:], axis=1)
        denominator = (np.linalg.norm(seq[:-1], axis=1) * np.linalg.norm(seq[1:], axis=1)) + 1e-8
        consecutive = numerator / denominator
        feats.extend([float(np.mean(consecutive)), float(np.min(consecutive)), float(np.std(consecutive))])
    else:
        feats.extend([0.0, 0.0, 0.0])

    return np.nan_to_num(np.asarray(feats, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def _train_order_classifier(
    frame: pd.DataFrame,
    sequences: list[list[int]],
    *,
    arc: str,
    random_state: int,
    shuffles_per_mix: int,
) -> ArcModelArtifact | None:
    """Torch-free learned arc tier: logistic regression on real vs shuffled order."""
    embeddings, embedding_source = compute_track_embeddings(frame)
    rng = np.random.default_rng(int(random_state))

    rows: list[np.ndarray] = []
    labels: list[float] = []
    for sequence in sequences:
        if len(sequence) < 3:
            continue
        rows.append(_order_summary_features(frame, sequence, embeddings))
        labels.append(1.0)
        for _ in range(int(shuffles_per_mix)):
            shuffled = list(sequence)
            rng.shuffle(shuffled)
            rows.append(_order_summary_features(frame, shuffled, embeddings))
            labels.append(0.0)

    if len(rows) < 4 or len(set(labels)) < 2:
        return None

    features = np.vstack(rows)
    targets = np.asarray(labels, dtype=int)
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced')),
    ])
    pipeline.fit(features, targets)

    summary = {
        'backend': 'order_clf',
        'mixes': len(sequences),
        'samples': int(len(targets)),
        'shuffles_per_mix': int(shuffles_per_mix),
        'embedding_source': embedding_source,
    }
    logger.info(
        'Trained torch-free order arc model on %d mixes (%d samples, %d features).',
        len(sequences), len(targets), int(features.shape[1]),
    )
    return ArcModelArtifact(
        backend='order_clf', arc=arc, order_model=pipeline,
        feature_dim=int(features.shape[1]), training_summary=summary,
    )


def train_arc_model(
    training_df: pd.DataFrame,
    *,
    arc: str = DEFAULT_ARC,
    random_state: int = 42,
    epochs: int = 40,
    shuffles_per_mix: int = _ORDER_SHUFFLES_PER_MIX,
) -> ArcModelArtifact:
    """Train the best available arc tier; falls back to the heuristic arc fit.

    Tiers, in order of preference: a PyTorch GRU when torch is present and there
    are enough mixes; otherwise a torch-free order classifier (low-data); else the
    heuristic.
    """
    frame = add_sentiment_features(training_df.reset_index(drop=True))
    sequences = _reconstruct_mix_sequences(frame)

    if _torch_available() and len(sequences) >= 4:
        return _train_torch_arc_model(frame, sequences, arc=arc, random_state=random_state, epochs=epochs)

    if len(sequences) >= _ORDER_MIN_MIXES:
        order_model = _train_order_classifier(
            frame, sequences, arc=arc, random_state=random_state, shuffles_per_mix=shuffles_per_mix,
        )
        if order_model is not None:
            return order_model

    logger.info('Too few reconstructable mixes (%d) for a learned arc model; using heuristic.', len(sequences))
    return heuristic_arc_model(arc)


def _train_torch_arc_model(
    frame: pd.DataFrame,
    sequences: list[list[int]],
    *,
    arc: str,
    random_state: int,
    epochs: int,
) -> ArcModelArtifact:
    import torch
    from torch import nn

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
