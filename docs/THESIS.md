# Music as Sequenced Affect: Directional, Set-Aware, Cross-Modal Music Selection and Mixing

> **Status: working draft / skeleton.** Prose is concise on purpose. Sections marked
> `[AGENT TODO]` need a follow-up agent to run experiments, fill tables, or write
> citations. Every chapter cites the code that implements it so the completing
> agent can navigate `mai/` directly. Keep the concise register; expand only where
> a `[AGENT TODO]` asks.

---

## Abstract

Most music-information-retrieval (MIR) similarity is **symmetric, static, and
single-modal**: how alike are two tracks. Real listening contexts are not. A DJ
asks a *directional* question (does A hand off into B), over a *set* (does this
track belong in the build / peak / cool), and increasingly a *cross-modal* one
(what scores this image / scene). This thesis proposes a single organising idea —
**a compact, interpretable affect space (valence, arousal, tension, warmth) shared
across audio, image, and text** — and builds four contributions on top of it: (C1)
a directional transition-craft model trained with hard-negative mining; (C2)
interpretable cross-modal scene→music grounding; (C3) set-level narrative-arc
modelling from order-as-supervision; and (C4) DJ-style *overlay* transitions that
mix mid-song by spectral complementarity. A staged retrieval funnel makes the
system interactive at million-track scale. `[AGENT TODO: 1-sentence headline result
once MAI-Bench is run.]`

**Thesis statement.** Music selection and mixing are best modelled as operations in
a shared, interpretable affect space: directionally between tracks, globally over a
set, and across modalities from a conditioning scene.

---

## 1. Introduction

**Problem.** Picking and sequencing music well is three problems the literature
usually treats separately: (i) which two tracks *transition* well (directional);
(ii) what whole *ordering* tells a story (set-level); (iii) what music fits a
non-musical *context* like a film scene (cross-modal). A common representation lets
one system address all three.

**Approach.** Project every entity — a track's edges, a whole track, an image, a
caption — into the same four affect axes (`mai/sentiment.py`,
`mai/scene_features.py`, `mai/scene_context.py`). Then transitions, arcs, and
scene-matching become geometry + craft features in that space.

**Contributions.**
- **C1** Directional transition model with hard-negative mining and train/serve-
  symmetric musical features (`mai/transition_model.py`, `mai/hard_negatives.py`,
  `mai/musical_features.py`).
- **C2** Interpretable image/scene→music grounding; a controllable affect space as
  an alternative to opaque CLAP embeddings (`mai/scene_*.py`, `mai/affect_probe.py`,
  `mai/cross_modal_model.py`).
- **C3** Narrative-arc set model learned from real-vs-shuffled order
  (`mai/sequence_model.py`).
- **C4** Overlay transitions: mid-song mixing by spectral complementarity, with a
  cue-sheet mix planner (`mai/overlay.py`, `mai/mix_planner.py`).
- **C5 (systems)** Staged retrieval funnel for interactive latency at scale
  (`mai/scene_index.py`).
- **Artifact** MAI-Bench: a scene→music benchmark from film cue sheets
  (`mai/scene_dataset.py`, `mai/scene_eval.py`).

---

## 2. Related Work `[AGENT TODO: write + cite]`

Position against, with real citations:
- **Joint audio-language embeddings** — CLAP, MuLan, MusicLM/MuLan retrieval. Gap:
  text↔audio, not image/scene↔audio, and opaque (not controllable).
- **Automatic DJ / beat-mixing** — work on beatmatching, cue-point detection,
  Automatic-DJ systems. Gap: tempo/beat focus; little affective or learned *craft*,
  no spectral-complementarity overlay scoring.
- **Playlist / sequence modelling** — similarity-graph and session-based methods.
  Gap: symmetric & static; no directional craft, no narrative arc.
- **Music emotion recognition** — DEAM, PMEmo, Emotify, MTG-Jamendo mood. Use these
  to *validate* the affect axes (§9).
`[AGENT TODO: 1 short paragraph each, 3-6 refs per bucket; a comparison table
(modality, directionality, interpretability, sequencing) vs CLAP/MuLan/Auto-DJ.]`

---

## 3. The Shared Affect Space (framework)

Four axes in [0,1]: **valence, arousal, tension, warmth** (`SENTIMENT_DIMS`).

- **Audio →** derived from acoustic descriptors (`mai/sentiment.py::add_sentiment_features`):
  e.g. arousal from energy/onset/tempo/brightness; tension from roughness/onset/
  (1−valence)/(1−harmonic). Edge-aware (`intro_`/`outro_`).
- **Image →** colour/light statistics → axes (`mai/scene_features.py`): warmth from
  warm-hue share + saturation; valence from brightness/saturation/warm hue; arousal
  from saturation/contrast/colourfulness; tension from contrast/darkness/coolness.
- **Text →** emotion lexicon anchors averaged by matched cues (`mai/scene_context.py`).

Why interpretable axes over a black-box embedding: they are **controllable** (a user
can steer "more tension"), **auditable**, and **modality-bridging** by construction.
`mai/affect_probe.py` shows any frozen embedding (CLAP/CLIP/MERT) can be *distilled*
into these axes via a closed-form ridge probe whose weights are the explanation —
keeping the interpretability while inheriting backbone quality.

`[AGENT TODO: formalise. Define the axis maps f_audio, f_image, f_text; state the
matching objective (weighted distance / Gaussian kernel in §5); note invariances.]`

---

## 4. C1 — Directional Transition Craft

A handoff A→B is **directional**: outro of A into intro of B. Model
(`mai/transition_model.py`):
- **Hard negatives** (`mai/hard_negatives.py`): for a positive A→B, keep A, swap in
  C from B's neighbourhood — forces the classifier onto the *plausible-but-unchosen*
  boundary, not the trivial similarity boundary.
- **Musical interaction features** (`mai/musical_features.py`): Camelot/circle-of-
  fifths, octave-folded tempo, energy/arousal/tension trajectories. **Computed by one
  primitive for both training rows and the N×N scoring matrix** → train/serve
  symmetric by construction. (A real train/serve column-order skew bug was found and
  fixed — see commit `02f2a2e`; documents why the invariant matters.)
- **Low-data tiers**: kNN backend when positives are scarce, RandomForest when ample;
  CV-AUC gate; adaptive shrinkage of the learned-model blend weight.

`[AGENT TODO: results — CV-AUC vs #positives curve; ablate hard vs random negatives
on held-out DJ-set transitions (1001tracklists / mixesDB ground truth).]`

---

## 5. C2 — Cross-Modal Scene→Music

Pipeline (`mai/scene_match.py`): image mood ⊕ text mood (blended by `text_weight`)
→ rank library by a Gaussian on axis-weighted mood distance + a genre-hint boost.
Both image and text land in §3's space, so matching is geometry in four numbers.

- **Scoring**: `score = exp(-d²/2σ²)`, `d` = axis-weighted distance; genre boost is a
  small multiplicative re-rank. Vectorised (no per-row loop).
- **Interpretable probe** (`mai/affect_probe.py`): distil backbone → axes.
- **Learned upgrade** (`mai/cross_modal_model.py`): CLIP-style symmetric-InfoNCE
  image↔music joint space, trained on film-cue pairs (torch-guarded seam).
- **Generation dual** (`mai/scene_generation.py`): when nothing fits, `scene_to_prompt`
  → MusicGen.

`[AGENT TODO: results — P@k/MRR/nDCG on MAI-Bench vs baselines (random, genre-only,
CLAP zero-shot); ablate colour-only / text-only / no-genre-boost; show interpretable
≈ CLAP + controllability.]`

---

## 6. C3 — Set-Level Narrative Arc

`mai/sequence_model.py`: an energy/arousal **arc fit** against a target shape
(rise→peak→cool) is always on; a **torch-free order classifier** learns real-vs-
shuffled ordering from ~11 order-summary scalars (order-as-supervision); an optional
GRU when torch + enough mixes. Ordering = beam search → 2-opt → arc orientation
(`mai/playlist_generation.py`).

`[AGENT TODO: results — does the learned arc score separate real DJ-set orderings
from shuffles (AUC)? human preference of arc-oriented vs greedy orderings.]`

---

## 7. C4 — Overlay Transitions (mid-song mixing)

`mai/overlay.py`. A transition is a **region overlay** `(exit_A, entry_B, offset,
blend_type, score)`, entry possibly mid-song.
- **Spectral complementarity** (novel): overlay well = fill *different* frequency
  pockets, `1 − Σ min(bandA_i, bandB_i)`; rewards A-bass-under-B-highs over lookalikes.
- **Score**: tempo lock + Camelot key + complementarity + hard vocal-clash penalty +
  phrase alignment + energy continuity.
- **Beat alignment**: FFT onset cross-correlation (O(n log n)).
- **Blend selection**: long_blend / bass_swap / double_drop / echo_out / loop_roll / cut.
- **Segmentation**: Foote novelty (`foote_novelty_boundaries`), Numba-`njit` seam.
- **Bridge/export**: `mai/mix_planner.py` builds regions from track rows → `plan_mix`
  → cue-sheet CSV.

`[AGENT TODO: validate against recorded DJ mixes — mine the overlap zone between
consecutive tracks as positive region-pairs (self-supervision), score them vs random
region pairs; small DJ user study on blend-type appropriateness.]`

---

## 8. C5 — Systems: Retrieval Funnel at Scale

`mai/scene_index.py`. Never score 1M per query: `filter (metadata) → ANN recall
(mood space) → exact rerank → diversify (MMR or k-DPP) → order`. Optional seams
(Polars, hnswlib/usearch/Faiss-IVFPQ, int8, Parquet) with exact numpy fallbacks;
`query_batch` is one BLAS matmul; incremental `add`.

- **Complexity**: per query ~O(F) filter + O(log N) ANN + O(k) rerank, k≪N.
- **Target**: ~10 ms/query at 1e6 (filter→ANN→rerank→diversify), single core.

`[AGENT TODO: measure — recall@k vs exact, and latency vs N (1e4..1e6) for brute vs
HNSW vs Faiss-PQ; memory under quantization ladder.]`

---

## 9. Evaluation Methodology & MAI-Bench

**Benchmark** (`mai/scene_dataset.py`): scene→track ground truth from **film/TV cue
sheets** (`ingest_cue_sheets`); JSONL schema + validator; `make_synthetic_benchmark`
for harness checks only (not real labels).
**Metrics** (`mai/scene_eval.py`): P@k, R@k, MRR, MAP, nDCG; bootstrap CIs;
**paired bootstrap significance**.
**Baselines**: random, genre-lexicon, CLAP zero-shot; **ablation flags** on the Mai
retriever.
**Affect validation**: regress predicted axes on DEAM/PMEmo human valence/arousal
(`mai/affect_probe.py`) → R² per axis.

`[AGENT TODO — the core empirical work:]`
1. Collect a real cue-sheet dump (Tunefind / IMDb soundtrack timings / cue sheets);
   run `ingest_cue_sheets` → MAI-Bench; report dataset stats.
2. Run `compare_retrievers` + `paired_bootstrap_test`; fill Table 1.
3. Affect-axis validation vs DEAM/PMEmo; fill Table 2.
4. Ablations (colour/text/genre); fill Table 3.
5. Multiple seeds, CIs, significance throughout.

---

## 10. Results `[AGENT TODO: fill from §9]`

Placeholder — current evidence is a **synthetic** harness check only (labels
fabricated from the model's own ranking, so it validates the *pipeline*, not
quality): mai-affect P@1=1.00, MAP=1.00; vs random MRR Δ≈+0.81, 95% CI [+0.47,+1.00],
p≈0. **Not a scientific result** — replace with real-cue-sheet numbers.

- Table 1: scene→music retrieval (MAI-Bench) — Mai vs baselines. `[TODO]`
- Table 2: affect-axis R² vs DEAM/PMEmo. `[TODO]`
- Table 3: ablations. `[TODO]`
- Table 4: transition-model AUC, hard vs random negatives. `[TODO]`
- Table 5: funnel recall@k / latency vs N. `[TODO]`

---

## 11. Deployment & Impact `[AGENT TODO: expand]`

Content-creation auto-soundtrack (Reels/TikTok/CapCut); NLE plugins
(Premiere/DaVinci) suggesting sync music for a cut; DJ-software craft-aware
transition + overlay suggestions (cue sheet → Rekordbox/Traktor); a HuggingFace
Space demo (image → playlist). `[TODO: pick one, build a thin demo, report usage.]`

---

## 12. Limitations, Ethics, Future Work

- **Affect subjectivity** → anchor to human MER datasets (§9).
- **Heuristic provenance** → must show learned ≥ heuristic on MAI-Bench, or
  interpretable ≈ black-box + controllability.
- **Licensing/ethics** → use cue-sheet *metadata* + features, never redistribute
  audio; document provenance and consent for any user study.
- **Future**: contrastive image↔music at scale; retrieval⊕generation; preference
  learning (RLHF) from DJ A/B; raw-audio CLAP/MERT encoders behind the existing seams.

---

## Appendix A — Code → Chapter Map

| Chapter | Modules |
|---|---|
| §3 affect space | `sentiment.py`, `scene_features.py`, `scene_context.py`, `affect_probe.py` |
| §4 transition | `transition_model.py`, `hard_negatives.py`, `musical_features.py` |
| §5 scene→music | `scene_match.py`, `scene_features.py`, `scene_context.py`, `cross_modal_model.py`, `scene_generation.py` |
| §6 arc | `sequence_model.py`, `playlist_generation.py` |
| §7 overlay | `overlay.py`, `mix_planner.py`, `beat_align.py` |
| §8 funnel | `scene_index.py` |
| §9 eval | `scene_eval.py`, `scene_dataset.py` |

## Appendix B — Reproducibility

Python 3.11+. `pip install -r requirements.txt`. Tests: `pytest tests/`. Scene/overlay
modules degrade to numpy/pandas/Pillow when sklearn/torch/hnswlib/polars are absent
(graceful seams). Build env note: agent shell has no `python` on PATH; a codex
runtime python (numpy/pandas/Pillow, no sklearn/torch) runs the lightweight subset
via a fake-`mai`-package isolation trick (see project memory `mai-build-environment`).

## Appendix C — Completing-Agent Checklist

- [ ] §2 related work + citations + comparison table.
- [ ] §3 formal definitions of the axis maps and matching objective.
- [ ] §9.1 collect real cue sheets → MAI-Bench; dataset stats.
- [ ] §9.2–9.4 run benchmarks/ablations/affect-validation; fill Tables 1–5.
- [ ] §4/§6/§7 empirical validations (transition AUC, arc AUC, overlay self-supervision).
- [ ] §11 one deployable demo.
- [ ] Polish abstract headline result; multiple seeds + significance everywhere.
