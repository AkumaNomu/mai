import json
from pathlib import Path
from typing import Any

import numpy as np


def _safe_array(value: Any) -> np.ndarray:
    return np.array(value, dtype=object)


class EmbeddingCache:
    def __init__(self, path: Path, backend: str, version: int, embedding_dim: int) -> None:
        self.path = path
        self.backend = backend
        self.version = version
        self.embedding_dim = embedding_dim
        self.keys: list[str] = []
        self.mtimes: list[float] = []
        self.embeddings: list[np.ndarray] = []
        self.tag_probs: list[np.ndarray] = []
        self.tags: list[str] = []

        if path.exists():
            try:
                data = np.load(path, allow_pickle=True)
                backend_val = str(data["backend"][0])
                version_val = int(data["version"][0])
                dim_val = int(data["embedding_dim"][0])
                if backend_val == backend and version_val == version and dim_val == embedding_dim:
                    self.keys = data["keys"].tolist()
                    self.mtimes = data["mtimes"].tolist()
                    self.embeddings = [np.array(v) for v in data["embeddings"]]
                    self.tag_probs = [np.array(v) for v in data["tag_probs"]]
                    self.tags = data["tags"].tolist()
            except Exception:
                self.keys = []
                self.mtimes = []
                self.embeddings = []
                self.tag_probs = []
                self.tags = []

    def _index(self, key: str) -> int | None:
        try:
            return self.keys.index(key)
        except ValueError:
            return None

    def get_or_compute(self, path: Path, mtime: float, backend: str) -> dict[str, Any] | None:
        key = str(path.resolve()).replace("\\", "/")
        idx = self._index(key)
        if idx is not None and float(self.mtimes[idx]) == float(mtime):
            return {
                "embedding": self.embeddings[idx],
                "tag_probs": self.tag_probs[idx],
                "tags": self.tags,
            }

        embedding, tag_probs, tags = extract_embedding(path, backend)
        if embedding is None or tag_probs is None or not tags:
            return None

        if not self.tags:
            self.tags = tags
        elif self.tags != tags:
            # Tags mismatch, reset cache to avoid inconsistent ordering
            self.keys = []
            self.mtimes = []
            self.embeddings = []
            self.tag_probs = []
            self.tags = tags
            idx = None

        if idx is None:
            self.keys.append(key)
            self.mtimes.append(float(mtime))
            self.embeddings.append(embedding)
            self.tag_probs.append(tag_probs)
        else:
            self.mtimes[idx] = float(mtime)
            self.embeddings[idx] = embedding
            self.tag_probs[idx] = tag_probs

        return {
            "embedding": embedding,
            "tag_probs": tag_probs,
            "tags": tags,
        }

    def save(self) -> None:
        if not self.keys:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.path,
            keys=_safe_array(self.keys),
            mtimes=np.array(self.mtimes, dtype=float),
            embeddings=np.stack(self.embeddings, axis=0),
            tag_probs=np.stack(self.tag_probs, axis=0),
            tags=_safe_array(self.tags),
            backend=_safe_array([self.backend]),
            version=np.array([self.version], dtype=int),
            embedding_dim=np.array([self.embedding_dim], dtype=int),
        )


def extract_embedding(path: Path, backend: str) -> tuple[np.ndarray | None, np.ndarray | None, list[str]]:
    if backend != "musicnn":
        raise ValueError("Only backend='musicnn' is supported in v1.")

    try:
        from musicnn.extractor import extractor as musicnn_extractor
    except Exception:
        return None, None, []

    taggram, tags, features = musicnn_extractor(str(path), model="MSD_musicnn")
    if taggram is None or tags is None:
        return None, None, []

    tag_probs = np.mean(taggram, axis=0)
    embedding = None
    if isinstance(features, dict):
        if "mean" in features:
            embedding = np.array(features["mean"])
        elif "embedding" in features:
            embedding = np.array(features["embedding"])
    elif isinstance(features, np.ndarray):
        if features.ndim == 2:
            embedding = np.mean(features, axis=0)
        else:
            embedding = features

    if embedding is None:
        embedding = tag_probs

    return np.array(embedding, dtype=float), np.array(tag_probs, dtype=float), list(tags)


def load_genre_map(path: Path | None) -> dict[str, list[str]]:
    if not path or not path.exists():
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        return {}
    return {str(k): list(v) for k, v in obj.items() if isinstance(v, list)}


def build_audio_genre_dist(
    tag_probs: np.ndarray,
    tags: list[str],
    genre_map_path: Path | None,
) -> dict[str, float]:
    genre_map = load_genre_map(genre_map_path)
    if not genre_map or tag_probs is None or not tags:
        return {}

    scores: dict[str, float] = {}
    for tag, prob in zip(tags, tag_probs):
        mapped = genre_map.get(tag, [])
        if not mapped:
            continue
        for genre in mapped:
            scores[genre] = scores.get(genre, 0.0) + float(prob)

    total = sum(scores.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in scores.items()}
