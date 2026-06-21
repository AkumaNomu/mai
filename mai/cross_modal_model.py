"""Contrastive image↔music model: a CLIP for scene→song.

The lexicon + colour baseline grounds a scene in a hand-built affect space. The
learned upgrade is to train a joint embedding directly from real scene↔track
pairs (film cue sheets), CLIP-style: pull a scene and its cued track together,
push mismatched pairs apart (symmetric InfoNCE). At inference a scene embeds into
the shared space and ranks tracks by cosine.

The model is *encoder-agnostic*: it consumes precomputed per-item feature vectors
— image features from CLIP/SigLIP, music features from the descriptor embedding or
MERT/CLAP — and learns lightweight projection heads on top. That keeps training
cheap and lets the heavy encoders stay optional.

PyTorch is required to *train or run* this model and is imported lazily, so the
module imports cleanly without torch; callers fall back to the affect baseline
(:mod:`mai.scene_match`) and the interpretable probe (:mod:`mai.affect_probe`).
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


@dataclass(slots=True)
class CrossModalPayload:
    """Serialisable weights for the two projection heads + training metadata."""

    image_dim: int
    music_dim: int
    joint_dim: int
    state_dict: dict[str, np.ndarray] = field(default_factory=dict)
    training_summary: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str) -> None:
        with open(path, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str) -> 'CrossModalPayload':
        with open(path, 'rb') as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, CrossModalPayload):
            raise TypeError(f'unexpected payload type: {type(payload)!r}')
        return payload


def _build_projection(nn, in_dim: int, joint_dim: int):
    return nn.Sequential(
        nn.Linear(int(in_dim), int(joint_dim)),
        nn.GELU(),
        nn.Linear(int(joint_dim), int(joint_dim)),
    )


def _import_torch():
    try:
        import torch
        from torch import nn
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            'The contrastive image↔music model requires PyTorch. Install torch and retry; '
            'the affect baseline (mai.scene_match) runs without it.'
        ) from exc
    return torch, nn


def train_image_music(
    image_features: np.ndarray,
    music_features: np.ndarray,
    *,
    joint_dim: int = 128,
    epochs: int = 100,
    batch_size: int = 256,
    temperature: float = 0.07,
    learning_rate: float = 1e-3,
    random_state: int = 42,
) -> CrossModalPayload:  # pragma: no cover - optional heavy path
    """Train symmetric InfoNCE over paired (image, music) feature rows.

    ``image_features[i]`` and ``music_features[i]`` are a matched scene↔track pair.
    Returns projection-head weights that embed either modality into a shared space.
    """
    torch, nn = _import_torch()
    if len(image_features) != len(music_features):
        raise ValueError('image_features and music_features must have equal length (paired rows)')
    if len(image_features) < 2:
        raise ValueError('contrastive training needs at least two pairs')

    torch.manual_seed(int(random_state))
    image_dim = int(image_features.shape[1])
    music_dim = int(music_features.shape[1])

    image_head = _build_projection(nn, image_dim, joint_dim)
    music_head = _build_projection(nn, music_dim, joint_dim)
    log_temp = nn.Parameter(torch.tensor(float(np.log(1.0 / temperature))))
    params = list(image_head.parameters()) + list(music_head.parameters()) + [log_temp]
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=1e-4)

    image_tensor = torch.as_tensor(np.asarray(image_features), dtype=torch.float32)
    music_tensor = torch.as_tensor(np.asarray(music_features), dtype=torch.float32)
    n = len(image_tensor)
    rng = np.random.default_rng(int(random_state))

    last_loss = float('nan')
    for _ in range(int(epochs)):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            if len(idx) < 2:
                continue
            batch_idx = torch.as_tensor(idx, dtype=torch.long)
            img = nn.functional.normalize(image_head(image_tensor[batch_idx]), dim=1)
            mus = nn.functional.normalize(music_head(music_tensor[batch_idx]), dim=1)
            scale = torch.exp(log_temp)
            logits = scale * img @ mus.t()
            labels = torch.arange(len(idx))
            loss = 0.5 * (nn.functional.cross_entropy(logits, labels)
                          + nn.functional.cross_entropy(logits.t(), labels))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu())

    state = {f'image.{k}': v.detach().cpu().numpy() for k, v in image_head.state_dict().items()}
    state.update({f'music.{k}': v.detach().cpu().numpy() for k, v in music_head.state_dict().items()})
    state['log_temp'] = log_temp.detach().cpu().numpy()
    summary = {'pairs': int(n), 'epochs': int(epochs), 'final_loss': last_loss, 'joint_dim': int(joint_dim)}
    logger.info('Trained contrastive image↔music model on %d pairs (final loss=%.4f).', n, last_loss)
    return CrossModalPayload(
        image_dim=image_dim, music_dim=music_dim, joint_dim=int(joint_dim),
        state_dict=state, training_summary=summary,
    )


def _load_head(torch, nn, payload: CrossModalPayload, prefix: str, in_dim: int):
    head = _build_projection(nn, in_dim, payload.joint_dim)
    head.load_state_dict({
        key[len(prefix) + 1:]: torch.as_tensor(value)
        for key, value in payload.state_dict.items() if key.startswith(prefix + '.')
    })
    head.eval()
    return head


def embed_images(payload: CrossModalPayload, image_features: np.ndarray) -> np.ndarray:  # pragma: no cover
    """Project image features into the joint space (L2-normalised rows)."""
    torch, nn = _import_torch()
    head = _load_head(torch, nn, payload, 'image', payload.image_dim)
    with torch.no_grad():
        out = nn.functional.normalize(head(torch.as_tensor(np.atleast_2d(image_features), dtype=torch.float32)), dim=1)
    return out.cpu().numpy()


def embed_music(payload: CrossModalPayload, music_features: np.ndarray) -> np.ndarray:  # pragma: no cover
    """Project music features into the joint space (L2-normalised rows)."""
    torch, nn = _import_torch()
    head = _load_head(torch, nn, payload, 'music', payload.music_dim)
    with torch.no_grad():
        out = nn.functional.normalize(head(torch.as_tensor(np.atleast_2d(music_features), dtype=torch.float32)), dim=1)
    return out.cpu().numpy()


def rank_music_for_image(
    payload: CrossModalPayload,
    image_feature: np.ndarray,
    music_features: np.ndarray,
    top_k: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover - optional heavy path
    """Rank a music feature bank against one image. Returns (indices, scores)."""
    image_joint = embed_images(payload, np.atleast_2d(image_feature))
    music_joint = embed_music(payload, music_features)
    scores = (music_joint @ image_joint[0])
    order = np.argsort(-scores)
    if top_k is not None:
        order = order[:top_k]
    return order, scores[order]
