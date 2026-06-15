# Mai Architecture

A concise map of how Mai turns a pool of songs into seamlessly flowing,
optimally ordered playlists — and the reasoning behind each modelling choice.

## Pipeline at a glance

```
ingest ─▶ audio analysis ─▶ sentiment ─▶ genre resolution
        ─▶ representation (descriptor / pretrained embedding)
        ─▶ directed transition matrix  (heuristic components + learned model)
        ─▶ ordering  (beam search → 2-opt refine → arc orientation)
        ─▶ export
```

Two model artifacts back the intelligence, both trained by `python -m mai.train`
from the scraper's positive-transition CSV:

| Artifact | File | Role |
|----------|------|------|
| Transition model | `data/cache/transition_model.joblib` | "Do these two tracks hand off well?" (pairwise) |
| Arc model | `data/cache/arc_model.joblib` | "Is this whole ordering a real set?" (sequence) |

## Module responsibilities

- `musical_features.py` — domain pairwise features: Camelot/circle-of-fifths
  harmonic compatibility, octave-aware tempo lock, energy/mood trajectory. Shared
  by the scorer and the model so training features == serving features.
- `hard_negatives.py` — mines close-but-unchosen negatives from real positives.
- `embeddings.py` — per-track vector (descriptor embedding always on; pretrained
  MERT/CLAP encoder seam behind `MAI_AUDIO_ENCODER`).
- `transition_model.py` — supervised pairwise scorer (kNN / RandomForest / torch
  MLP, the CPU family auto-chosen by data size), trained on positives + hard
  negatives, reporting cross-validated AUC and a shrunk recommended blend weight.
- `beat_align.py` — beat-grid / phrase compatibility at the splice point.
- `cross_modal.py` — mood-continuity grounding that rewards seamless cross-genre
  jumps (CLAP upgrade seam).
- `sequence_model.py` — set-level arc model: energy/arousal arc fit always on; a
  torch-free order classifier (real vs shuffled order) as the low-data learned
  tier; an optional GRU over track embeddings when torch and enough mixes exist.
- `playlist_generation.py` — directed transition matrix + beam search + 2-opt
  refinement + arc orientation.

## Improvements (thesis form)

Each entry: **problem → method → why it works**.

### 1. Hard-negative mining replaces random negatives
**Problem.** The scraper yields only positive (DJ-chosen) handoffs; the prior
model paired each with a uniformly random cross-mix track. Random negatives
differ in genre/tempo/key, so the classifier learns the trivial "are these songs
similar?" boundary rather than transition craft.
**Method.** For each positive A→B, retain the outgoing A and substitute an
incoming C drawn from B's nearest neighbours in feature space, excluding B itself
and same-mix tracks. A minority of easy random negatives is kept for probability
calibration (`hard_negatives.mine_hard_negatives`).
**Why.** Training pressure concentrates on the decision boundary between a good
handoff and a *plausible but unchosen* one — the only boundary that matters at
inference, where the generator is always choosing among near neighbours.

### 2. Musical interaction features (train/serve symmetric)
**Problem.** Pairwise features were generic scalar deltas; harmonic mixing and
octave-equivalent tempo matching — the core of DJ craft — were invisible to the
model.
**Method.** Camelot-wheel compatibility, circle-of-fifths distance, relative
major/minor detection, octave-folded tempo agreement, and directional
energy/arousal/tension trajectories. The same primitive computes both the
training row features and the N×N scoring matrix (verified identical to machine
precision), eliminating train/serve skew.
**Why.** Encoding the invariances a human DJ uses (key relationships, half/double
time) lets a small model generalise from few labels.

### 3. Cross-validated AUC as a data-quality gate
**Problem.** Training reported only row counts, hiding whether the data actually
separates good from bad handoffs.
**Method.** Stratified k-fold ROC-AUC computed at train time and stored in the
artifact summary.
**Why.** Makes data quality observable before any playlist is generated; a low
AUC signals scrape/label problems, not a generation bug.

### 4. Beat-grid & phrase alignment
**Problem.** Transition scoring ignored beatmatchability and phrase landing.
**Method.** A directed outro→intro score from octave-aware tempo lock, joint
beat-grid stability, and downbeat handoff strength — relaxed when the outgoing
tail leaves silence to drop into (`beat_align`). Absolute beat *phase* is left as
a documented raw-audio hook.
**Why.** Tempo-locked, downbeat-aligned cuts are the difference between a blend
and a bump; this rewards them explicitly.

### 5. Cross-modal mood grounding for cross-genre flow
**Problem.** Genre balancing nudged the generator toward genre islands and
penalised genre jumps, fighting the goal of seamless multi-genre sets.
**Method.** Each edge is projected into a mood space (valence/arousal/tension/
warmth + brightness/energy); the directed mood similarity is *lifted* when a
genre boundary is crossed with the mood intact (`cross_modal`). A CLAP joint
audio/text space is the documented upgrade.
**Why.** Listeners perceive bridges emotionally, not categorically. Rewarding
mood continuity makes a pop→electronic jump that *feels* right score high.

### 6. Set-level arc modelling
**Problem.** Pairwise scores cannot express global shape — why a track belongs
*now* in the build/peak/cool of a set.
**Method.** An energy/arousal arc fit against a target arc (default
rise→peak→cool), always available; plus an optional GRU trained to distinguish
real mix orderings from shuffles (`sequence_model`).
**Why.** Captures the narrative of a DJ set that no local pair score can see.

### 7. Near-optimal ordering: beam → 2-opt → arc orientation
**Problem.** Beam search is greedy-with-lookahead and leaves locally poor
sub-orderings; pure greedy could not recover.
**Method.** After beam search, a bounded 2-opt sweep raises total directed flow,
then the path is oriented (forward vs reverse) to best fit the energy arc
(`playlist_generation`). Bounded by length/passes for predictable runtime.
**Why.** 2-opt removes the dominant local-search regret cheaply; arc orientation
aligns the refined order with set narrative.

### 8. Runtime optimisation
**Problem.** Per-step candidate ranking sorted Python lists; costly on large pools.
**Method.** Vectorised `argpartition`/`argsort` top-k candidate selection in the
beam, refinement caps, and shared cached representations.
**Why.** Quality features add work; vectorising the hot path keeps end-to-end
runtime flat or better.

### 9. Scrape data-quality lever
**Problem.** Resolution acceptance thresholds were hardcoded magic numbers, so
training-data precision could not be tuned.
**Method.** `MAI_RESOLUTION_MIN_SCORE` / `MAI_RESOLUTION_LOW_OVERLAP_MIN_*`
env-tunable gates at the candidate-selection point (`training_scrape`).
**Why.** Cleaner positives directly raise the transition model's ceiling;
precision/recall is now a dial, not a code edit.

### 10. Low-data regime: order self-supervision, kNN, adaptive shrinkage
**Problem.** Curating scraped DJ mixes yields tens — not thousands — of labelled
handoffs. A deep pairwise classifier and a GRU both overfit at that scale, and a
fixed learned-model weight lets a noisy model overpower the heuristics.
**Method.** Three sample-efficient choices. (a) A *torch-free order classifier*
(`sequence_model._train_order_classifier`): each real mix order is contrasted
against many shuffles, described by ~11 order-summary scalars (arc shape,
roughness, embedding smoothness), and separated by logistic regression — the
always-on learned arc tier, with the GRU reserved for when torch and ample mixes
are both present. (b) A *kNN transition backend* auto-selected when positives are
scarce (`transition_model._resolve_estimator`), since distance-weighted
neighbours generalise without deep trees to overfit. (c) *Adaptive shrinkage*
(`recommend_transition_model_weight`): the suggested blend weight scales with CV
AUC above chance and with the positive count, and is 0 when AUC is unavailable.
**Why.** The order of any curated playlist is free supervision — no transition
labelling needed — so the learned signal grows with every scrape, while the
heuristic core (sections 2, 4, 5) keeps quality high from the very first run.

## Training and use

```powershell
python -m mai.train --train-csv data/training/positive_transitions.csv
python run.py --csv data/Playlist.csv `
  --transition-model-path data/cache/transition_model.joblib `
  --transition-model-weight 0.35
```

Pretrained encoders (MERT/CLAP) and the learned arc/transition GRU activate only
when PyTorch (and, for CLAP, `transformers`) are installed; the descriptor
representation, arc fit, and RandomForest scorer are the always-on baseline.
