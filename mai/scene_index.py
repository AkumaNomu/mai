"""Scale-out scene→music retrieval: a staged funnel over a large track pool.

Scoring a million tracks per query is wasteful. This module precomputes a track
index once, then narrows the pool in coarse→fine stages so the expensive scene
scorer only ever touches a few thousand survivors:

    hard filter (metadata)  ->  ANN recall (mood space)
        ->  exact rerank (scene_match)  ->  MMR diversify  ->  optional order

Every stage has a heavy-dep fast path and a pure numpy/pandas fallback, so the
funnel runs anywhere and only gets faster when the optional libraries are present:

* **filter**  — Polars (predicate pushdown) if installed, else pandas masks.
* **ANN**     — hnswlib / usearch if installed, else a vectorised numpy brute
                scan (exact; trivial in the 4-D mood space even at 1e6).
* **rerank**  — the exact :func:`mai.scene_match.score_library_against_scene`.
* **diversify** — Maximal Marginal Relevance over mood vectors.

The index is picklable (numpy arrays + a pandas filter frame) and supports
incremental ``add`` so new songs do not force a full rebuild.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .scene_match import (
    SCENE_MOOD_DIMS,
    SceneMoodTarget,
    score_library_against_scene,
)
from .sentiment import add_sentiment_features


logger = logging.getLogger(__name__)

# Descriptor columns used for the (optional) high-dimensional embedding space.
# Kept local so importing this module never drags in sklearn (embeddings.py does).
_EMBEDDING_BASES = (
    'tempo', 'energy', 'danceability', 'loudness', 'acousticness', 'speechiness',
    'liveness', 'valence', 'spectral_centroid', 'spectral_bandwidth',
    'spectral_rolloff', 'spectral_flatness', 'zcr', 'onset_strength', 'harmonic_ratio',
)
_FILTER_COLUMNS = ('genre_primary', 'mix_group', 'style_cluster', 'tempo', 'key', 'mode')
_ID_CANDIDATES = ('track_id', 'video_id', 'id', 'resolved_video_id')


def _robust_scale(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median/IQR robust scaling (sklearn-free), returning (scaled, center, scale)."""
    center = np.median(matrix, axis=0)
    q75, q25 = np.percentile(matrix, [75, 25], axis=0)
    scale = np.where((q75 - q25) > 1e-9, q75 - q25, 1.0)
    return (matrix - center) / scale, center, scale


def _resolve_id_column(library: pd.DataFrame) -> str | None:
    return next((c for c in _ID_CANDIDATES if c in library.columns), None)


def _mood_matrix(library: pd.DataFrame) -> np.ndarray:
    enriched = add_sentiment_features(library)
    arrays = []
    for dim in SCENE_MOOD_DIMS:
        column = f'sentiment_{dim}'
        if column in enriched.columns:
            arrays.append(pd.to_numeric(enriched[column], errors='coerce').fillna(0.5).to_numpy(dtype=np.float64))
        else:
            arrays.append(np.full(len(enriched), 0.5))
    return np.column_stack(arrays).astype(np.float32)


def _embedding_matrix(library: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    columns = [base for base in _EMBEDDING_BASES if base in library.columns]
    if not columns:
        return np.zeros((len(library), 0), dtype=np.float32), np.zeros(0), np.ones(0)
    raw = library[columns].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=np.float64)
    scaled, center, scale = _robust_scale(raw) if len(library) > 1 else (raw, np.zeros(raw.shape[1]), np.ones(raw.shape[1]))
    return np.nan_to_num(scaled).astype(np.float32), center, scale


@dataclass(slots=True)
class _AnnIndex:
    """Uniform wrapper over whichever ANN backend is installed.

    Exposes a single ``query(vector, k) -> labels`` regardless of backend, so the
    recall stage is backend-agnostic. ``backend == 'brute'`` means no native ANN
    library was found and the exact numpy scan in the recall stage is used instead.
    """

    backend: str
    handle: Any = None

    def query(self, vector: np.ndarray, k: int) -> np.ndarray:
        q = np.ascontiguousarray(vector[None, :], dtype=np.float32)
        if self.backend == 'hnswlib':
            labels, _ = self.handle.knn_query(q, k=int(k))
            return labels[0].astype(int)
        if self.backend == 'usearch':
            matches = self.handle.search(q, int(k))
            return np.asarray(matches.keys, dtype=int).reshape(-1)
        if self.backend == 'faiss':
            _, labels = self.handle.search(q, int(k))
            return labels[0].astype(int)
        raise RuntimeError(f'ANN backend {self.backend!r} has no native query')


def _build_ann(matrix: np.ndarray) -> _AnnIndex:
    """Build the best available ANN index; fall back to an exact brute scan."""
    n, dim = matrix.shape
    if dim == 0 or n < 100:  # tiny pools: brute is already instant and exact
        return _AnnIndex(backend='brute')
    data = np.ascontiguousarray(matrix, dtype=np.float32)

    try:
        import hnswlib
        index = hnswlib.Index(space='l2', dim=int(dim))
        index.init_index(max_elements=n, ef_construction=200, M=16)
        index.add_items(data, np.arange(n))
        index.set_ef(128)
        return _AnnIndex(backend='hnswlib', handle=index)
    except Exception:
        pass

    try:
        from usearch.index import Index as USearchIndex
        index = USearchIndex(ndim=int(dim), metric='l2sq')
        index.add(np.arange(n), data)
        return _AnnIndex(backend='usearch', handle=index)
    except Exception:
        pass

    try:  # Faiss IVF-PQ: billion-scale, GPU-capable, ~32x compression via PQ.
        import faiss
        cells = int(max(1, min(4096, round(np.sqrt(n)))))
        quantizer = faiss.IndexFlatL2(int(dim))
        sub = int(max(1, min(dim // 2, 32)))
        if dim >= 8 and n >= 10000:
            index = faiss.IndexIVFPQ(quantizer, int(dim), cells, sub, 8)
            index.train(data)
        else:
            index = faiss.IndexFlatL2(int(dim))
        index.add(data)
        if hasattr(index, 'nprobe'):
            index.nprobe = max(1, cells // 16)
        return _AnnIndex(backend='faiss', handle=index)
    except Exception:
        pass

    return _AnnIndex(backend='brute')


def _polars_available() -> bool:
    try:
        import polars  # noqa: F401
    except Exception:
        return False
    return True


def _polars_filter(filter_frame: pd.DataFrame, filters: dict[str, Any]) -> np.ndarray:
    """Polars predicate-pushdown filter, parity-matched to the pandas mask path.

    Lazy columnar evaluation over Arrow buffers; same case-insensitive string and
    numeric min/max semantics as :meth:`SceneIndex._filter_stage`'s pandas branch.
    """
    import polars as pl

    frame = pl.from_pandas(filter_frame.reset_index(drop=True)).with_row_index('__row__')
    expr = pl.lit(True)
    for column, predicate in filters.items():
        if column not in filter_frame.columns:
            continue
        lowered = pl.col(column).cast(pl.Utf8).str.strip_chars().str.to_lowercase()
        numeric = pl.col(column).cast(pl.Float64, strict=False)
        if isinstance(predicate, dict):
            if 'in' in predicate:
                allowed = [str(v).strip().lower() for v in predicate['in']]
                expr = expr & lowered.is_in(allowed)
            if 'min' in predicate:
                expr = expr & (numeric >= float(predicate['min'])).fill_null(False)
            if 'max' in predicate:
                expr = expr & (numeric <= float(predicate['max'])).fill_null(False)
        else:
            allowed = ([str(predicate).strip().lower()] if isinstance(predicate, str)
                       else [str(v).strip().lower() for v in predicate])
            expr = expr & lowered.is_in(allowed)
    rows = frame.filter(expr).get_column('__row__').to_numpy()
    return np.asarray(rows, dtype=int)


def quantize_int8(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-column int8 quantization for a fast, low-memory coarse distance pass.

    Returns ``(codes, lo, scale)`` such that ``codes * scale + lo ≈ matrix``. 4x
    smaller than f32 and SIMD-friendly; used only as a coarse prefilter, never for
    the exact rerank (the scene scorer always reruns in float).
    """
    lo = matrix.min(axis=0)
    hi = matrix.max(axis=0)
    scale = np.where((hi - lo) > 1e-12, (hi - lo) / 255.0, 1.0)
    codes = np.clip(np.round((matrix - lo) / scale), 0, 255).astype(np.uint8)
    return codes, lo.astype(np.float32), scale.astype(np.float32)


@dataclass(slots=True)
class SceneIndex:
    """Precomputed, queryable index over a track library."""

    track_ids: list[Any]
    mood_matrix: np.ndarray                       # N x 4 (valence/arousal/tension/warmth)
    embedding_matrix: np.ndarray                  # N x D robust-scaled descriptors (may be N x 0)
    filter_frame: pd.DataFrame                    # metadata columns for predicate filtering
    library: pd.DataFrame                         # original rows, index-aligned
    embedding_center: np.ndarray = field(default_factory=lambda: np.zeros(0))
    embedding_scale: np.ndarray = field(default_factory=lambda: np.ones(0))
    ann_space: str = 'mood'
    _ann: Any = field(default=None, repr=False, compare=False)

    @property
    def size(self) -> int:
        return len(self.track_ids)

    # ----- construction ---------------------------------------------------- #
    @classmethod
    def build(cls, library: pd.DataFrame, *, ann_space: str = 'mood') -> 'SceneIndex':
        library = library.reset_index(drop=True)
        id_column = _resolve_id_column(library)
        track_ids = (library[id_column].tolist() if id_column else list(range(len(library))))

        mood = _mood_matrix(library)
        embedding, center, scale = _embedding_matrix(library)
        filter_frame = library[[c for c in _FILTER_COLUMNS if c in library.columns]].copy()

        index = cls(
            track_ids=track_ids, mood_matrix=mood, embedding_matrix=embedding,
            filter_frame=filter_frame, library=library, embedding_center=center,
            embedding_scale=scale, ann_space=ann_space if embedding.shape[1] else 'mood',
        )
        index._rebuild_ann()
        logger.info('Built SceneIndex over %d tracks (ann_space=%s, dim=%d).',
                    index.size, index.ann_space, index._ann_matrix().shape[1])
        return index

    def _ann_matrix(self) -> np.ndarray:
        return self.embedding_matrix if self.ann_space == 'embedding' else self.mood_matrix

    def _rebuild_ann(self) -> None:
        self._ann = _build_ann(self._ann_matrix())

    def add(self, library: pd.DataFrame) -> None:
        """Append new tracks. Reuses the stored embedding scaling for consistency."""
        new = library.reset_index(drop=True)
        id_column = _resolve_id_column(new)
        new_ids = (new[id_column].tolist() if id_column else
                   list(range(self.size, self.size + len(new))))
        new_mood = _mood_matrix(new)
        if self.embedding_matrix.shape[1] and len(self.embedding_scale):
            columns = [b for b in _EMBEDDING_BASES if b in new.columns]
            raw = new[columns].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=np.float64)
            new_embed = np.nan_to_num((raw - self.embedding_center) / self.embedding_scale).astype(np.float32)
        else:
            new_embed = np.zeros((len(new), self.embedding_matrix.shape[1]), dtype=np.float32)

        self.track_ids.extend(new_ids)
        self.mood_matrix = np.vstack([self.mood_matrix, new_mood])
        if self.embedding_matrix.shape[1]:
            self.embedding_matrix = np.vstack([self.embedding_matrix, new_embed])
        self.filter_frame = pd.concat(
            [self.filter_frame, new[[c for c in self.filter_frame.columns if c in new.columns]]],
            ignore_index=True,
        )
        self.library = pd.concat([self.library, new], ignore_index=True)
        self._rebuild_ann()

    # ----- query stages ---------------------------------------------------- #
    def _filter_stage(self, filters: dict[str, Any] | None) -> np.ndarray:
        n = self.size
        if not filters:
            return np.arange(n)
        if _polars_available():
            try:
                return _polars_filter(self.filter_frame, filters)
            except Exception:
                logger.exception('Polars filter failed; using pandas masks.')
        mask = np.ones(n, dtype=bool)
        for column, predicate in filters.items():
            if column not in self.filter_frame.columns:
                logger.warning('Filter column %r not in index; ignoring.', column)
                continue
            series = self.filter_frame[column]
            if isinstance(predicate, dict):
                if 'in' in predicate:
                    allowed = {str(v).strip().lower() for v in predicate['in']}
                    mask &= series.astype(str).str.strip().str.lower().isin(allowed).to_numpy()
                if 'min' in predicate:
                    mask &= (pd.to_numeric(series, errors='coerce') >= float(predicate['min'])).fillna(False).to_numpy()
                if 'max' in predicate:
                    mask &= (pd.to_numeric(series, errors='coerce') <= float(predicate['max'])).fillna(False).to_numpy()
            else:  # scalar / iterable equality
                allowed = ({str(predicate).strip().lower()} if isinstance(predicate, str)
                           else {str(v).strip().lower() for v in predicate})
                mask &= series.astype(str).str.strip().str.lower().isin(allowed).to_numpy()
        return np.nonzero(mask)[0]

    def _recall_stage(self, target: SceneMoodTarget, candidates: np.ndarray, recall_k: int) -> np.ndarray:
        if len(candidates) <= recall_k:
            return candidates
        query = target.mood_vector().astype(np.float32)
        if self.ann_space == 'embedding':
            # No scene→embedding map yet; fall back to mood ANN for the scene query.
            query_space = self.mood_matrix
        else:
            query_space = self.mood_matrix

        # Native ANN only valid when the index covers the full pool (filters would
        # otherwise let it return rows that were filtered out).
        ann = self._ann
        if (ann is not None and ann.backend != 'brute'
                and self.ann_space == 'mood' and len(candidates) == self.size):
            try:
                return ann.query(query, int(min(recall_k, self.size)))
            except Exception:
                logger.exception('ANN query failed; using brute scan.')
        # Brute vectorised L2 over candidate rows (exact). 4-D mood => cheap at 1e6.
        sub = query_space[candidates]
        dist = np.sum((sub - query[None, :]) ** 2, axis=1)
        keep = np.argpartition(dist, int(recall_k))[:recall_k]
        return candidates[keep]

    def query(
        self,
        target: SceneMoodTarget,
        *,
        filters: dict[str, Any] | None = None,
        recall_k: int = 2000,
        top_k: int = 25,
        diversify: bool | str = True,
        mmr_lambda: float = 0.6,
        order: bool = False,
    ) -> pd.DataFrame:
        """Run the funnel and return the top tracks for ``target``, best first.

        ``diversify`` accepts ``True``/``'mmr'`` (Maximal Marginal Relevance),
        ``'dpp'`` (k-DPP greedy MAP), or ``False``/``'none'`` (pure scene-fit).
        """
        if self.size == 0:
            return self.library.copy()

        candidates = self._filter_stage(filters)
        if len(candidates) == 0:
            logger.info('No tracks passed the filter stage.')
            return self.library.head(0).copy()

        recalled = self._recall_stage(target, candidates, recall_k)
        subframe = self.library.iloc[recalled].reset_index(drop=True)

        reranked = score_library_against_scene(subframe, target, top_k=None)
        mode = 'mmr' if diversify is True else (diversify or 'none')
        if mode != 'none' and len(reranked) > top_k:
            if mode == 'dpp':
                reranked = _dpp_select(reranked, top_k=top_k)
            else:
                reranked = _mmr_select(reranked, top_k=top_k, lam=mmr_lambda)
        else:
            reranked = reranked.head(top_k).reset_index(drop=True)

        if order and len(reranked) >= 3:
            from .scene_match import order_scene_playlist
            reranked = order_scene_playlist(reranked, target)
        return reranked

    def query_batch(
        self,
        targets: Sequence[SceneMoodTarget],
        *,
        top_k: int = 25,
        recall_k: int = 2000,
    ) -> list[pd.DataFrame]:
        """Score many scenes at once.

        The recall distances for all queries are one ``(Q x N)`` matmul (BLAS),
        amortising the scan across the batch — the shape that maps onto a GPU when
        throughput, not single-query latency, is the goal.
        """
        if not targets:
            return []
        mood = self.mood_matrix.astype(np.float64)            # N x 4
        queries = np.vstack([t.mood_vector() for t in targets])  # Q x 4
        # ||n - q||^2 = ||n||^2 - 2 n·q + ||q||^2, computed for all (q, n) at once.
        nn = np.sum(mood ** 2, axis=1)[None, :]
        qq = np.sum(queries ** 2, axis=1)[:, None]
        dist = qq + nn - 2.0 * (queries @ mood.T)             # Q x N
        keep = min(recall_k, self.size)
        results = []
        for row, target in enumerate(targets):
            recalled = np.argpartition(dist[row], keep - 1)[:keep] if keep < self.size else np.arange(self.size)
            subframe = self.library.iloc[recalled].reset_index(drop=True)
            results.append(score_library_against_scene(subframe, target, top_k=top_k))
        return results

    # ----- persistence ----------------------------------------------------- #
    def save_parquet(self, directory: str) -> None:
        """Persist columnar (Arrow/Parquet) for the frames + ``.npz`` for arrays.

        Parquet is mmap-friendly and language-agnostic; falls back to :meth:`save`
        (single pickle) when pyarrow is unavailable.
        """
        try:
            import pyarrow  # noqa: F401
        except Exception:
            logger.info('pyarrow not installed; using pickle persistence.')
            self.save(os.path.join(directory, 'scene_index.pkl') if os.path.isdir(directory) else directory)
            return
        os.makedirs(directory, exist_ok=True)
        self.library.to_parquet(os.path.join(directory, 'library.parquet'))
        self.filter_frame.to_parquet(os.path.join(directory, 'filter.parquet'))
        np.savez(
            os.path.join(directory, 'arrays.npz'),
            mood=self.mood_matrix, embedding=self.embedding_matrix,
            center=self.embedding_center, scale=self.embedding_scale,
            track_ids=np.asarray(self.track_ids, dtype=object), ann_space=np.asarray(self.ann_space),
        )

    @classmethod
    def load_parquet(cls, directory: str) -> 'SceneIndex':
        arrays = np.load(os.path.join(directory, 'arrays.npz'), allow_pickle=True)
        index = cls(
            track_ids=list(arrays['track_ids']), mood_matrix=arrays['mood'],
            embedding_matrix=arrays['embedding'], embedding_center=arrays['center'],
            embedding_scale=arrays['scale'], ann_space=str(arrays['ann_space']),
            filter_frame=pd.read_parquet(os.path.join(directory, 'filter.parquet')),
            library=pd.read_parquet(os.path.join(directory, 'library.parquet')),
        )
        index._rebuild_ann()
        return index

    def save(self, path: str) -> None:
        state = {
            'track_ids': self.track_ids, 'mood_matrix': self.mood_matrix,
            'embedding_matrix': self.embedding_matrix, 'filter_frame': self.filter_frame,
            'library': self.library, 'embedding_center': self.embedding_center,
            'embedding_scale': self.embedding_scale, 'ann_space': self.ann_space,
        }
        with open(path, 'wb') as handle:
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> 'SceneIndex':
        with open(path, 'rb') as handle:
            state = pickle.load(handle)
        index = cls(
            track_ids=state['track_ids'], mood_matrix=state['mood_matrix'],
            embedding_matrix=state['embedding_matrix'], filter_frame=state['filter_frame'],
            library=state['library'], embedding_center=state['embedding_center'],
            embedding_scale=state['embedding_scale'], ann_space=state['ann_space'],
        )
        index._rebuild_ann()
        return index


def _mmr_select(reranked: pd.DataFrame, *, top_k: int, lam: float) -> pd.DataFrame:
    """Maximal Marginal Relevance over mood vectors: relevance vs redundancy.

    ``reranked`` must carry ``scene_fit`` and ``sentiment_*`` columns (the scene
    scorer adds the former; the latter come from ``add_sentiment_features``).
    Picks a diverse, high-fit subset so the result is not 12 near-identical songs.
    """
    lam = float(np.clip(lam, 0.0, 1.0))
    enriched = add_sentiment_features(reranked)
    mood = np.column_stack([
        pd.to_numeric(enriched.get(f'sentiment_{d}', 0.5), errors='coerce').fillna(0.5).to_numpy()
        for d in SCENE_MOOD_DIMS
    ]).astype(np.float64)
    relevance = pd.to_numeric(reranked['scene_fit'], errors='coerce').fillna(0.0).to_numpy()

    norms = mood / (np.linalg.norm(mood, axis=1, keepdims=True) + 1e-9)
    sim = np.clip(norms @ norms.T, 0.0, 1.0)  # mood cosine similarity, [0, 1]

    n = len(reranked)
    selected: list[int] = []
    chosen_mask = np.zeros(n, dtype=bool)
    # Running max similarity of each candidate to the already-selected set; updated
    # in one vectorised max per pick instead of an inner python loop over selected.
    max_sim_to_selected = np.zeros(n, dtype=np.float64)
    for step in range(min(top_k, n)):
        if step == 0:
            choice = int(np.argmax(relevance))
        else:
            mmr = lam * relevance - (1.0 - lam) * max_sim_to_selected
            mmr[chosen_mask] = -np.inf
            choice = int(np.argmax(mmr))
        selected.append(choice)
        chosen_mask[choice] = True
        max_sim_to_selected = np.maximum(max_sim_to_selected, sim[:, choice])
    return reranked.iloc[selected].reset_index(drop=True)


def _dpp_select(reranked: pd.DataFrame, *, top_k: int, quality_weight: float = 1.0) -> pd.DataFrame:
    """Greedy MAP inference for a k-DPP: diverse-and-relevant subset selection.

    Builds an L-ensemble kernel ``L = q_i q_j <phi_i, phi_j>`` (quality-weighted
    mood similarity) and greedily adds the item with the largest marginal gain via
    an incremental Cholesky — the standard fast DPP-MAP. More principled than MMR
    when redundancy is higher-order (a song redundant with a *pair* already chosen).
    """
    enriched = add_sentiment_features(reranked)
    mood = np.column_stack([
        pd.to_numeric(enriched.get(f'sentiment_{d}', 0.5), errors='coerce').fillna(0.5).to_numpy()
        for d in SCENE_MOOD_DIMS
    ]).astype(np.float64)
    relevance = pd.to_numeric(reranked['scene_fit'], errors='coerce').fillna(0.0).to_numpy()

    norms = mood / (np.linalg.norm(mood, axis=1, keepdims=True) + 1e-9)
    quality = np.exp(quality_weight * relevance)  # q_i; larger fit => more mass
    kernel = (quality[:, None] * quality[None, :]) * np.clip(norms @ norms.T, 0.0, 1.0)

    n = len(reranked)
    k = min(top_k, n)
    selected: list[int] = []
    cholesky = np.zeros((k, n))                       # rows = picks so far
    diag = np.diag(kernel).astype(np.float64).copy()  # remaining marginal gains
    for step in range(k):
        candidate = int(np.argmax(np.where(diag > 1e-12, diag, -np.inf)))
        if diag[candidate] <= 1e-12:
            break
        if step > 0:
            c_candidate = cholesky[:step, candidate]                # accumulated e's at the new pick
            cross = cholesky[:step, :]                              # step x n
            updates = (kernel[candidate, :] - c_candidate @ cross) / np.sqrt(diag[candidate])
        else:
            updates = kernel[candidate, :] / np.sqrt(diag[candidate])
        cholesky[step, :] = updates
        diag = diag - updates ** 2
        diag[candidate] = -np.inf
        selected.append(candidate)
    return reranked.iloc[selected].reset_index(drop=True)


def build_scene_index(library: pd.DataFrame, *, ann_space: str = 'mood') -> SceneIndex:
    """Convenience wrapper: build a :class:`SceneIndex` from a track library."""
    return SceneIndex.build(library, ann_space=ann_space)


def query_scene_index(
    index: SceneIndex,
    target: SceneMoodTarget,
    *,
    filters: dict[str, Any] | None = None,
    recall_k: int = 2000,
    top_k: int = 25,
    diversify: bool | str = True,
    order: bool = False,
) -> pd.DataFrame:
    """Functional alias for :meth:`SceneIndex.query` (matches the funnel spec)."""
    return index.query(
        target, filters=filters, recall_k=recall_k, top_k=top_k,
        diversify=diversify, order=order,
    )
