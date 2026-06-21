"""Interpretable affect probe: distil any embedding into the affect axes.

The cross-modal moat is *interpretability*. A black-box CLAP/CLIP embedding can
rank songs but cannot say *why* — and cannot be steered. This module fits a small
linear (ridge) probe from any fixed embedding (descriptor, MERT, CLAP, CLIP) onto
the four named affect axes ``valence / arousal / tension / warmth``.

Linear-on-frozen-features is deliberate: it is the standard interpretability tool
(a "probe"), it is cheap and closed-form (ridge regression has an exact solution),
it needs no torch, and the learned weight matrix is itself the explanation — each
axis is a readable linear combination of embedding dimensions. Predicted axes plug
straight into the existing scene scorer, so a pretrained backbone can drive Mai
while keeping the controllable, human-readable affect space.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass

import numpy as np


logger = logging.getLogger(__name__)

AFFECT_AXES = ('valence', 'arousal', 'tension', 'warmth')


@dataclass(slots=True)
class AffectProbe:
    """A fitted linear map ``embedding -> affect axes`` with standardisation."""

    weight: np.ndarray            # (D+1) x A, last row is the bias
    feature_mean: np.ndarray      # D
    feature_scale: np.ndarray     # D
    axes: tuple[str, ...] = AFFECT_AXES
    train_r2: dict[str, float] | None = None

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict affect axes (clipped to [0, 1]) for a batch of embeddings."""
        x = np.atleast_2d(np.asarray(embeddings, dtype=np.float64))
        standardized = (x - self.feature_mean) / self.feature_scale
        augmented = np.column_stack([standardized, np.ones(len(standardized))])
        return np.clip(augmented @ self.weight, 0.0, 1.0)

    def predict_dict(self, embedding: np.ndarray) -> dict[str, float]:
        values = self.predict(embedding)[0]
        return {axis: float(values[i]) for i, axis in enumerate(self.axes)}

    def explain(self, axis: str, embedding_names: list[str], top: int = 8) -> list[tuple[str, float]]:
        """Largest-magnitude embedding dimensions driving one affect axis."""
        if axis not in self.axes:
            raise ValueError(f'unknown axis {axis!r}; have {self.axes}')
        column = self.weight[:-1, self.axes.index(axis)]
        order = np.argsort(-np.abs(column))[:top]
        return [(embedding_names[i] if i < len(embedding_names) else f'dim_{i}', float(column[i])) for i in order]

    def save(self, path: str) -> None:
        with open(path, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str) -> 'AffectProbe':
        with open(path, 'rb') as handle:
            probe = pickle.load(handle)
        if not isinstance(probe, AffectProbe):
            raise TypeError(f'unexpected probe type: {type(probe)!r}')
        return probe


def _coefficient_of_determination(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    residual = np.sum((y_true - y_pred) ** 2, axis=0)
    total = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    return 1.0 - residual / np.where(total > 1e-12, total, 1.0)


def fit_affect_probe(
    embeddings: np.ndarray,
    affect_targets: np.ndarray,
    *,
    axes: tuple[str, ...] = AFFECT_AXES,
    l2: float = 1.0,
) -> AffectProbe:
    """Fit a ridge-regression probe mapping ``embeddings`` -> ``affect_targets``.

    ``affect_targets`` is ``(N, len(axes))`` in [0, 1] (e.g. DEAM/PMEmo human
    valence/arousal mapped onto the axes). Closed-form ridge solution; no torch.
    """
    x = np.asarray(embeddings, dtype=np.float64)
    y = np.asarray(affect_targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError('embeddings and affect_targets must both be 2-D')
    if len(x) != len(y):
        raise ValueError('embeddings and affect_targets must have the same number of rows')
    if y.shape[1] != len(axes):
        raise ValueError(f'affect_targets has {y.shape[1]} columns but {len(axes)} axes were given')

    feature_mean = np.mean(x, axis=0)
    feature_scale = np.std(x, axis=0)
    feature_scale = np.where(feature_scale > 1e-9, feature_scale, 1.0)
    standardized = (x - feature_mean) / feature_scale
    augmented = np.column_stack([standardized, np.ones(len(standardized))])

    d = augmented.shape[1]
    regulariser = l2 * np.eye(d)
    regulariser[-1, -1] = 0.0  # do not penalise the bias
    weight = np.linalg.solve(augmented.T @ augmented + regulariser, augmented.T @ y)

    predictions = np.clip(augmented @ weight, 0.0, 1.0)
    r2 = _coefficient_of_determination(y, predictions)
    train_r2 = {axis: float(r2[i]) for i, axis in enumerate(axes)}
    logger.info('Fitted affect probe on %d samples; train R^2=%s.', len(x),
                {k: round(v, 3) for k, v in train_r2.items()})
    return AffectProbe(
        weight=weight, feature_mean=feature_mean, feature_scale=feature_scale,
        axes=axes, train_r2=train_r2,
    )
