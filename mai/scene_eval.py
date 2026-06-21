"""Evaluation harness: turn the scene→music matcher into measurable science.

Engineering becomes a result when it is scored against ground truth and beaten
baselines. This module supplies:

* standard ranking metrics — precision@k, recall@k, MRR, MAP, nDCG@k;
* a :class:`Retriever` protocol so any method (Mai, random, genre-only, CLAP)
  plugs into the same harness;
* :func:`run_benchmark` which averages metrics over a set of labelled scenes.

Ground truth is a set of relevant track ids per scene (see :mod:`mai.scene_dataset`
for the MAI-Bench schema and the film-cue-sheet adapter). Optional graded
relevance enables nDCG; binary relevance is enough for the rest.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .scene_index import SceneIndex, _resolve_id_column
from .scene_match import build_scene_target, score_library_against_scene


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BenchmarkExample:
    """One labelled scene: its description plus the tracks that fit it."""

    scene_id: str
    relevant_ids: set[Any]
    image_path: str | None = None
    scene_text: str | None = None
    graded: dict[Any, float] = field(default_factory=dict)  # id -> relevance grade (for nDCG)


# --------------------------------------------------------------------------- #
# Ranking metrics. Each takes a ranked list of ids and the labelled example.   #
# --------------------------------------------------------------------------- #

def precision_at_k(ranked: list[Any], relevant: set[Any], k: int) -> float:
    if k <= 0:
        return 0.0
    top = ranked[:k]
    return sum(1 for item in top if item in relevant) / float(k)


def recall_at_k(ranked: list[Any], relevant: set[Any], k: int) -> float:
    if not relevant:
        return 0.0
    top = ranked[:k]
    return sum(1 for item in top if item in relevant) / float(len(relevant))


def reciprocal_rank(ranked: list[Any], relevant: set[Any]) -> float:
    for position, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / position
    return 0.0


def average_precision(ranked: list[Any], relevant: set[Any]) -> float:
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for position, item in enumerate(ranked, start=1):
        if item in relevant:
            hits += 1
            precision_sum += hits / float(position)
    return precision_sum / float(len(relevant))


def ndcg_at_k(ranked: list[Any], relevant: set[Any], k: int, graded: dict[Any, float] | None = None) -> float:
    grades = graded or {item: 1.0 for item in relevant}

    def dcg(items: list[Any]) -> float:
        return float(sum(grades.get(item, 0.0) / np.log2(position + 1)
                         for position, item in enumerate(items[:k], start=1)))

    actual = dcg(ranked)
    ideal_items = sorted(grades, key=lambda item: -grades[item])
    ideal = dcg(ideal_items)
    return actual / ideal if ideal > 1e-12 else 0.0


# --------------------------------------------------------------------------- #
# Retrievers                                                                   #
# --------------------------------------------------------------------------- #

class Retriever(Protocol):
    name: str

    def retrieve(self, example: BenchmarkExample, k: int) -> list[Any]:
        ...


@dataclass(slots=True)
class MaiAffectRetriever:
    """The Mai scene→music matcher (colour + context affect), the system under test."""

    library: pd.DataFrame
    index: SceneIndex | None = None
    text_weight: float = 0.5
    recall_k: int = 2000
    diversify: bool = False
    use_image: bool = True          # ablation: drop the colour/image signal
    use_text: bool = True           # ablation: drop the caption/context signal
    use_genre_boost: bool = True    # ablation: drop the genre-hint re-rank
    name: str = 'mai-affect'

    def retrieve(self, example: BenchmarkExample, k: int) -> list[Any]:
        image_path = example.image_path if self.use_image else None
        scene_text = example.scene_text if self.use_text else None
        if not image_path and not scene_text:  # ablation removed the only signal
            return []
        target = build_scene_target(image_path, scene_text, text_weight=self.text_weight)
        if not self.use_genre_boost and target.context is not None:
            target.context.genre_weights = {}
        if self.index is not None:
            matched = self.index.query(target, recall_k=self.recall_k, top_k=k, diversify=self.diversify)
            return _ids(matched, k)
        matched = score_library_against_scene(self.library, target, top_k=k)
        return _ids(matched, k)


@dataclass(slots=True)
class RandomRetriever:
    """Uniform-random ranking — the floor every real method must clear."""

    library: pd.DataFrame
    seed: int = 0
    name: str = 'random'

    def retrieve(self, example: BenchmarkExample, k: int) -> list[Any]:
        ids = _all_ids(self.library)
        rng = np.random.default_rng(abs(hash((self.seed, example.scene_id))) % (2 ** 32))
        order = rng.permutation(len(ids))
        return [ids[i] for i in order[:k]]


@dataclass(slots=True)
class GenreLexiconRetriever:
    """Caption genre hints only — isolates the text-genre signal from affect."""

    library: pd.DataFrame
    name: str = 'genre-lexicon'

    def retrieve(self, example: BenchmarkExample, k: int) -> list[Any]:
        from .scene_context import analyze_scene_text
        context = analyze_scene_text(example.scene_text)
        ids = _all_ids(self.library)
        column = next((c for c in ('genre_primary', 'mix_group', 'style_cluster') if c in self.library.columns), None)
        if not context.genre_weights or column is None:
            return ids[:k]
        labels = self.library[column].fillna('').astype(str).str.strip().str.lower().to_numpy()
        scores = np.array([
            max((w for g, w in context.genre_weights.items() if g in label or (label and label in g)), default=0.0)
            for label in labels
        ])
        order = np.argsort(-scores, kind='stable')
        return [ids[i] for i in order[:k]]


@dataclass(slots=True)
class ClapRetriever:
    """Zero-shot CLAP baseline: rank tracks by text→audio similarity to the caption.

    The reference learned baseline the affect model must beat or match-with-
    interpretability. Requires ``torch`` + ``transformers`` and ``MAI_CLAP_MODEL``;
    :meth:`available` reports whether it can run so the harness can skip it.
    """

    library: pd.DataFrame
    audio_features: np.ndarray | None = None  # precomputed CLAP audio embeddings, N x D
    name: str = 'clap-zeroshot'

    @staticmethod
    def available() -> bool:
        import os
        if not str(os.getenv('MAI_CLAP_MODEL') or '').strip():
            return False
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception:
            return False
        return True

    def retrieve(self, example: BenchmarkExample, k: int) -> list[Any]:  # pragma: no cover - heavy path
        import os
        import torch
        from transformers import ClapModel, ClapProcessor

        if self.audio_features is None:
            raise RuntimeError('ClapRetriever needs precomputed audio_features (N x D CLAP embeddings).')
        model_name = str(os.getenv('MAI_CLAP_MODEL'))
        model = ClapModel.from_pretrained(model_name)
        processor = ClapProcessor.from_pretrained(model_name)
        text = example.scene_text or ''
        inputs = processor(text=[text], return_tensors='pt', padding=True)
        with torch.no_grad():
            text_embed = model.get_text_features(**inputs).cpu().numpy()[0]
        audio = self.audio_features / (np.linalg.norm(self.audio_features, axis=1, keepdims=True) + 1e-9)
        scores = audio @ (text_embed / (np.linalg.norm(text_embed) + 1e-9))
        order = np.argsort(-scores)
        ids = _all_ids(self.library)
        return [ids[i] for i in order[:k]]


def _all_ids(library: pd.DataFrame) -> list[Any]:
    column = _resolve_id_column(library)
    return library[column].tolist() if column else list(range(len(library)))


def _ids(matched: pd.DataFrame, k: int) -> list[Any]:
    column = _resolve_id_column(matched)
    if column:
        return matched[column].head(k).tolist()
    return matched.head(k).index.tolist()


# --------------------------------------------------------------------------- #
# Benchmark runner                                                             #
# --------------------------------------------------------------------------- #

def evaluate_example(ranked: list[Any], example: BenchmarkExample, ks: tuple[int, ...]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f'P@{k}'] = precision_at_k(ranked, example.relevant_ids, k)
        metrics[f'R@{k}'] = recall_at_k(ranked, example.relevant_ids, k)
        metrics[f'nDCG@{k}'] = ndcg_at_k(ranked, example.relevant_ids, k, example.graded or None)
    metrics['MRR'] = reciprocal_rank(ranked, example.relevant_ids)
    metrics['MAP'] = average_precision(ranked, example.relevant_ids)
    return metrics


def run_benchmark(
    retriever: Retriever,
    examples: list[BenchmarkExample],
    *,
    ks: tuple[int, ...] = (1, 5, 10),
    retrieve_k: int | None = None,
) -> dict[str, float]:
    """Average ranking metrics for ``retriever`` over ``examples``."""
    if not examples:
        return {}
    depth = retrieve_k if retrieve_k is not None else max(ks)
    per_example: list[dict[str, float]] = []
    for example in examples:
        ranked = retriever.retrieve(example, depth)
        per_example.append(evaluate_example(ranked, example, ks))

    keys = per_example[0].keys()
    averaged = {key: float(np.mean([row[key] for row in per_example])) for key in keys}
    averaged['n_examples'] = float(len(examples))
    logger.info('Benchmark %s: %s', getattr(retriever, 'name', retriever.__class__.__name__),
                {k: round(v, 4) for k, v in averaged.items()})
    return averaged


def compare_retrievers(
    retrievers: list[Retriever],
    examples: list[BenchmarkExample],
    *,
    ks: tuple[int, ...] = (1, 5, 10),
) -> pd.DataFrame:
    """Run several retrievers on the same benchmark and tabulate the metrics."""
    rows = []
    for retriever in retrievers:
        metrics = run_benchmark(retriever, examples, ks=ks)
        metrics['retriever'] = getattr(retriever, 'name', retriever.__class__.__name__)
        rows.append(metrics)
    frame = pd.DataFrame(rows).set_index('retriever')
    ordered = [c for c in frame.columns if c != 'n_examples'] + ['n_examples']
    return frame[ordered]


def main(argv=None) -> int:
    """CLI: benchmark the scene retrievers on a MAI-Bench file (or a synthetic set)."""
    import argparse
    import logging

    parser = argparse.ArgumentParser(description='Benchmark scene→music retrievers.')
    parser.add_argument('--library', required=True, help='track library CSV.')
    parser.add_argument('--benchmark', help='MAI-Bench .jsonl/.csv; omit to use a synthetic set.')
    parser.add_argument('--ks', default='1,5,10', help='cutoffs, comma-separated (default 1,5,10).')
    parser.add_argument('--use-index', action='store_true', help='back the Mai retriever with a SceneIndex.')
    parser.add_argument('--significance', action='store_true', help='paired bootstrap: Mai vs random.')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format='%(message)s')

    library = pd.read_csv(args.library)
    library.columns = [c.strip() for c in library.columns]
    if args.benchmark:
        from .scene_dataset import load_benchmark
        examples = load_benchmark(args.benchmark)
    else:
        from .scene_dataset import make_synthetic_benchmark
        examples = make_synthetic_benchmark(library)
        print(f'No --benchmark given; using {len(examples)} SYNTHETIC scenes (harness check, not a real score).')

    ks = tuple(int(x) for x in str(args.ks).split(',') if x.strip())
    mai = MaiAffectRetriever(library=library, index=(SceneIndex.build(library) if args.use_index else None))
    retrievers = [mai, GenreLexiconRetriever(library=library), RandomRetriever(library=library)]
    if ClapRetriever.available():
        print('CLAP available but needs precomputed audio features; skipping in CLI.')

    table = compare_retrievers(retrievers, examples, ks=ks)
    print('\n' + table.round(4).to_string())

    if args.significance:
        result = paired_bootstrap_test(mai, RandomRetriever(library=library), examples, metric='MRR', k=max(ks))
        print(f'\nMai vs random (MRR@{max(ks)}): Δ={result["mean_diff"]:+.4f} '
              f'CI[{result["ci_lo"]:+.4f},{result["ci_hi"]:+.4f}] p(Mai≤random)={result["p_value_a_le_b"]:.4f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


# --------------------------------------------------------------------------- #
# Statistical rigor: per-example samples, bootstrap CIs, paired significance.  #
# --------------------------------------------------------------------------- #

def metric_samples(
    retriever: Retriever,
    examples: list[BenchmarkExample],
    *,
    metric: str = 'MRR',
    k: int = 10,
) -> np.ndarray:
    """Per-example value of one metric — the raw vector for CIs / significance."""
    ks = (k,) if metric not in ('MRR', 'MAP') else (k,)
    values = []
    for example in examples:
        ranked = retriever.retrieve(example, k)
        values.append(evaluate_example(ranked, example, ks)[metric])
    return np.asarray(values, dtype=np.float64)


def bootstrap_ci(
    values: np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean and a percentile bootstrap confidence interval ``(mean, lo, hi)``."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)


def paired_bootstrap_test(
    retriever_a: Retriever,
    retriever_b: Retriever,
    examples: list[BenchmarkExample],
    *,
    metric: str = 'MRR',
    k: int = 10,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Paired bootstrap: is ``retriever_a`` better than ``retriever_b`` on ``metric``?

    Resamples scenes (paired, so per-scene difficulty cancels) and reports the
    mean difference, its CI, and a one-sided p-value ``P(mean diff <= 0)``.
    """
    a = metric_samples(retriever_a, examples, metric=metric, k=k)
    b = metric_samples(retriever_b, examples, metric=metric, k=k)
    diff = a - b
    rng = np.random.default_rng(seed)
    boot = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    p_value = float(np.mean(boot <= 0.0))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        'metric': metric, 'mean_a': float(a.mean()), 'mean_b': float(b.mean()),
        'mean_diff': float(diff.mean()), 'ci_lo': float(lo), 'ci_hi': float(hi),
        'p_value_a_le_b': p_value, 'n': float(len(examples)),
    }
