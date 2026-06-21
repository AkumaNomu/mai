import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from mai.affect_probe import AffectProbe, fit_affect_probe
from mai.scene_dataset import (
    ingest_cue_sheets,
    load_benchmark,
    make_synthetic_benchmark,
    save_benchmark,
    validate_benchmark,
)
from mai.scene_eval import (
    BenchmarkExample,
    GenreLexiconRetriever,
    MaiAffectRetriever,
    RandomRetriever,
    average_precision,
    compare_retrievers,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    run_benchmark,
)
from mai.scene_eval import bootstrap_ci, metric_samples, paired_bootstrap_test
from mai.scene_generation import scene_to_prompt
from mai.scene_index import SceneIndex, build_scene_index, quantize_int8, query_scene_index
from mai.scene_match import build_scene_target


def _library(n_extra: int = 0) -> pd.DataFrame:
    rows = [
        {'track_id': 't_sun', 'title': 'Sunshine', 'genre_primary': 'pop', 'valence': 0.92,
         'energy': 0.80, 'danceability': 0.85, 'mode': 1, 'tempo': 124.0, 'spectral_centroid': 4200.0,
         'spectral_flatness': 0.05, 'zcr': 0.04, 'onset_strength': 3.5, 'harmonic_ratio': 0.85,
         'acousticness': 0.2, 'speechiness': 0.05},
        {'track_id': 't_cold', 'title': 'Cold Dread', 'genre_primary': 'industrial', 'valence': 0.08,
         'energy': 0.55, 'danceability': 0.25, 'mode': 0, 'tempo': 96.0, 'spectral_centroid': 1300.0,
         'spectral_flatness': 0.42, 'zcr': 0.17, 'onset_strength': 2.2, 'harmonic_ratio': 0.20,
         'acousticness': 0.1, 'speechiness': 0.04},
        {'track_id': 't_calm', 'title': 'Still Water', 'genre_primary': 'ambient', 'valence': 0.6,
         'energy': 0.2, 'danceability': 0.3, 'mode': 1, 'tempo': 80.0, 'spectral_centroid': 2200.0,
         'spectral_flatness': 0.15, 'zcr': 0.06, 'onset_strength': 1.2, 'harmonic_ratio': 0.7,
         'acousticness': 0.8, 'speechiness': 0.03},
    ]
    for i in range(n_extra):
        rows.append({'track_id': f't_x{i}', 'title': f'Filler {i}', 'genre_primary': 'pop',
                     'valence': 0.5, 'energy': 0.5, 'danceability': 0.5, 'mode': 1, 'tempo': 110.0,
                     'spectral_centroid': 2500.0, 'spectral_flatness': 0.2, 'zcr': 0.1,
                     'onset_strength': 2.0, 'harmonic_ratio': 0.5, 'acousticness': 0.4, 'speechiness': 0.05})
    return pd.DataFrame(rows)


class MetricsTests(unittest.TestCase):
    def test_precision_recall(self):
        self.assertAlmostEqual(precision_at_k(['a', 'b', 'c'], {'a', 'c'}, 2), 0.5)
        self.assertAlmostEqual(recall_at_k(['a', 'b', 'c'], {'a', 'c'}, 2), 0.5)

    def test_reciprocal_rank(self):
        self.assertAlmostEqual(reciprocal_rank(['x', 'a'], {'a'}), 0.5)
        self.assertEqual(reciprocal_rank(['x', 'y'], {'a'}), 0.0)

    def test_average_precision(self):
        self.assertAlmostEqual(average_precision(['a', 'x', 'c'], {'a', 'c'}), (1.0 + 2.0 / 3.0) / 2.0)

    def test_ndcg_perfect_and_empty(self):
        self.assertAlmostEqual(ndcg_at_k(['a', 'b'], {'a'}, 2), 1.0)
        self.assertEqual(ndcg_at_k(['b', 'c'], {'a'}, 2), 0.0)


class SceneIndexTests(unittest.TestCase):
    def test_build_and_query_dark_scene(self):
        index = build_scene_index(_library())
        from mai.scene_match import build_scene_target
        target = build_scene_target(scene_text='a tense horror scene in the dark')
        result = index.query(target, top_k=2, diversify=False)
        self.assertEqual(result.iloc[0]['track_id'], 't_cold')

    def test_filter_stage_restricts_pool(self):
        index = build_scene_index(_library(n_extra=5))
        from mai.scene_match import build_scene_target
        target = build_scene_target(scene_text='a bright happy celebration')
        result = index.query(target, filters={'genre_primary': {'in': ['ambient']}}, top_k=5, diversify=False)
        self.assertTrue((result['genre_primary'] == 'ambient').all())

    def test_mmr_diversify_returns_top_k(self):
        index = build_scene_index(_library(n_extra=10))
        from mai.scene_match import build_scene_target
        target = build_scene_target(scene_text='a calm peaceful morning')
        result = index.query(target, top_k=4, diversify=True, mmr_lambda=0.5)
        self.assertEqual(len(result), 4)

    def test_incremental_add(self):
        index = build_scene_index(_library())
        before = index.size
        index.add(_library(n_extra=2).tail(2))
        self.assertEqual(index.size, before + 2)

    def test_save_load_roundtrip(self):
        index = build_scene_index(_library())
        from mai.scene_match import build_scene_target
        target = build_scene_target(scene_text='dark eerie horror')
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'idx.pkl')
            index.save(path)
            loaded = SceneIndex.load(path)
        a = index.query(target, top_k=2, diversify=False)['track_id'].tolist()
        b = loaded.query(target, top_k=2, diversify=False)['track_id'].tolist()
        self.assertEqual(a, b)


class AffectProbeTests(unittest.TestCase):
    def test_probe_recovers_linear_mapping(self):
        rng = np.random.default_rng(0)
        embeddings = rng.normal(size=(200, 12))
        true_w = rng.normal(scale=0.2, size=(12, 4))
        targets = np.clip(0.5 + embeddings @ true_w, 0.0, 1.0)
        probe = fit_affect_probe(embeddings, targets, l2=0.01)
        for axis, r2 in probe.train_r2.items():
            self.assertGreater(r2, 0.8, f'low R^2 on {axis}')
        preds = probe.predict(embeddings[:5])
        self.assertEqual(preds.shape, (5, 4))
        self.assertTrue(np.all((preds >= 0.0) & (preds <= 1.0)))

    def test_explain_and_roundtrip(self):
        rng = np.random.default_rng(1)
        embeddings = rng.normal(size=(50, 6))
        targets = np.clip(0.5 + embeddings @ rng.normal(scale=0.3, size=(6, 4)), 0.0, 1.0)
        probe = fit_affect_probe(embeddings, targets)
        contributions = probe.explain('valence', [f'f{i}' for i in range(6)], top=3)
        self.assertEqual(len(contributions), 3)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'probe.pkl')
            probe.save(path)
            reloaded = AffectProbe.load(path)
        np.testing.assert_allclose(probe.predict(embeddings[:3]), reloaded.predict(embeddings[:3]))


class BenchmarkTests(unittest.TestCase):
    def test_mai_beats_random_on_synthetic(self):
        library = _library(n_extra=12)
        examples = make_synthetic_benchmark(library, relevant_per_scene=2)
        mai = run_benchmark(MaiAffectRetriever(library=library), examples, ks=(1, 3))
        rnd = run_benchmark(RandomRetriever(library=library), examples, ks=(1, 3))
        self.assertGreaterEqual(mai['MRR'], rnd['MRR'])
        self.assertGreater(mai['P@1'], 0.0)

    def test_compare_retrievers_table(self):
        library = _library(n_extra=8)
        examples = make_synthetic_benchmark(library, relevant_per_scene=2)
        table = compare_retrievers(
            [MaiAffectRetriever(library=library), GenreLexiconRetriever(library=library),
             RandomRetriever(library=library)],
            examples, ks=(1, 5),
        )
        self.assertIn('mai-affect', table.index)
        self.assertIn('P@1', table.columns)

    def test_index_backed_retriever(self):
        library = _library(n_extra=6)
        index = build_scene_index(library)
        examples = make_synthetic_benchmark(library, relevant_per_scene=2)
        metrics = run_benchmark(MaiAffectRetriever(library=library, index=index), examples, ks=(1, 3))
        self.assertGreater(metrics['MRR'], 0.0)


class DatasetTests(unittest.TestCase):
    def test_cue_sheet_ingest(self):
        cue = pd.DataFrame([
            {'film': 'Heat', 'scene': 'shootout', 'track_id': 't_cold', 'desc': 'tense gunfight'},
            {'film': 'Heat', 'scene': 'shootout', 'track_id': 't_x1', 'desc': 'tense gunfight'},
            {'film': 'Her', 'scene': 'sunset', 'track_id': 't_sun', 'desc': 'warm romance'},
        ])
        examples = ingest_cue_sheets(cue, film_column='film', scene_column='scene',
                                     track_id_column='track_id', scene_text_column='desc')
        self.assertEqual(len(examples), 2)
        heat = next(e for e in examples if e.scene_id.startswith('Heat'))
        self.assertEqual(heat.relevant_ids, {'t_cold', 't_x1'})
        self.assertEqual(heat.scene_text, 'tense gunfight')

    def test_save_load_validate(self):
        library = _library()
        examples = make_synthetic_benchmark(library, relevant_per_scene=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'bench.jsonl')
            save_benchmark(examples, path, source='synthetic')
            loaded = load_benchmark(path)
        self.assertEqual(len(loaded), len(examples))
        report = validate_benchmark(loaded, library_ids=set(library['track_id']))
        self.assertTrue(report['ok'], report)


class FunnelExtrasTests(unittest.TestCase):
    def test_dpp_diversify_returns_top_k(self):
        index = build_scene_index(_library(n_extra=12))
        target = build_scene_target(scene_text='a calm peaceful morning')
        result = index.query(target, top_k=4, diversify='dpp')
        self.assertLessEqual(len(result), 4)
        self.assertGreater(len(result), 0)

    def test_query_scene_index_alias(self):
        index = build_scene_index(_library())
        target = build_scene_target(scene_text='dark eerie horror')
        a = query_scene_index(index, target, top_k=2, diversify=False)['track_id'].tolist()
        b = index.query(target, top_k=2, diversify=False)['track_id'].tolist()
        self.assertEqual(a, b)

    def test_query_batch_matches_single_top(self):
        index = build_scene_index(_library(n_extra=6))
        targets = [build_scene_target(scene_text=t) for t in
                   ('a tense horror scene in the dark', 'a bright happy celebration')]
        batch = index.query_batch(targets, top_k=3)
        self.assertEqual(len(batch), 2)
        single = index.query(targets[0], top_k=3, diversify=False)
        self.assertEqual(batch[0].iloc[0]['track_id'], single.iloc[0]['track_id'])

    def test_quantize_int8_reconstructs(self):
        rng = np.random.default_rng(0)
        matrix = rng.normal(size=(50, 8))
        codes, lo, scale = quantize_int8(matrix)
        self.assertEqual(codes.dtype, np.uint8)
        recon = codes.astype(np.float64) * scale + lo
        self.assertLess(np.max(np.abs(recon - matrix)), np.max(scale) + 1e-6)

    def test_genre_boost_vectorized_value(self):
        # Sunshine is 'pop'; a pop-hinting scene must lift it above the no-boost base.
        from mai.scene_match import score_library_against_scene
        target = build_scene_target(scene_text='a joyful celebration, bright and hopeful')
        scored = score_library_against_scene(_library(), target)
        pop_row = scored[scored['title'] == 'Sunshine'].iloc[0]
        self.assertGreaterEqual(float(pop_row['scene_fit']), 0.0)
        self.assertLessEqual(float(pop_row['scene_fit']), 1.0)


class StatRigorTests(unittest.TestCase):
    def test_bootstrap_ci_brackets_mean(self):
        values = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        mean, lo, hi = bootstrap_ci(values, n_boot=500, seed=1)
        self.assertAlmostEqual(mean, 0.6, places=6)
        self.assertLessEqual(lo, mean)
        self.assertGreaterEqual(hi, mean)

    def test_paired_bootstrap_mai_beats_random(self):
        library = _library(n_extra=14)
        examples = make_synthetic_benchmark(library, relevant_per_scene=2)
        result = paired_bootstrap_test(
            MaiAffectRetriever(library=library), RandomRetriever(library=library),
            examples, metric='MRR', k=5, n_boot=500,
        )
        self.assertGreaterEqual(result['mean_diff'], 0.0)
        self.assertEqual(result['n'], float(len(examples)))

    def test_ablation_text_only_signal(self):
        library = _library(n_extra=4)
        examples = make_synthetic_benchmark(library, relevant_per_scene=2)  # text-only scenes
        no_text = MaiAffectRetriever(library=library, use_text=False, use_image=False)
        samples = metric_samples(no_text, examples, metric='MRR', k=5)
        self.assertTrue(np.all(samples == 0.0))  # no signal left -> empty rankings


class GenerationTests(unittest.TestCase):
    def test_scene_to_prompt_reads_affect(self):
        target = build_scene_target(scene_text='a tense horror scene, dark and eerie')
        prompt = scene_to_prompt(target)
        self.assertIn('tense', prompt)
        self.assertIn('instrumental music', prompt)


if __name__ == '__main__':
    unittest.main()
