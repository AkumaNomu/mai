# Mai
Transition-aware playlist tools for YouTube: reorder existing playlists, generate new ones, and scrape curated transitions for training data.

## Current Capabilities
- Pull a public YouTube playlist (metadata only) or read a CSV, normalize columns, and enrich with audio sentiment, key, tempo, and edge-intro/outro features.
- Score directed transitions and generate either a single reordered playlist or multiple fixed-size playlists with beam search, genre balancing, and optional reuse controls.
- Score handoffs with harmonic (Camelot/circle-of-fifths) compatibility, octave-aware tempo lock, beat/phrase alignment, and a cross-genre mood bridge so different genres flow seamlessly.
- Train a supervised transition model (hard-negative mining + musical interaction features, cross-validated AUC) and a set-level arc model with `python -m mai.train`.
- Refine ordering past beam search with a bounded 2-opt sweep and energy-arc orientation for near-optimal, narrative sets.
- Export the recommended order to YouTube Music or standard YouTube playlists.
- Scrape channel and video tracklists into labeled positive transitions, resolve tracks via search, analyze audio, and produce training pairs.
- Reusable caches for playlist metadata, audio features, search resolution, and training artifacts, plus a cleanup command.
- CLI progress renderer with heartbeat updates to avoid “silent” long steps.
- yt-dlp defaults use jsless extractor options; Windows gets an automatic Deno fallback for JS challenges.

## Setup
1. Python 3.11+ recommended (TOML parsing relies on `tomllib`).
2. Create a virtualenv and install deps:
   ```powershell
   python -m venv .venv
   & .venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```
3. Place auth files (paths can be set in `mai.toml`):
   - `data/youtube_client_secret.json`, `data/youtube_token.json` for standard YouTube exports.
   - `data/ytmusic_auth.json` for YouTube Music exports.

## Configuration (`mai.toml`)
Precedence: CLI flags > `mai.toml` > built-in defaults.
- `cache`: `root_dir`, `audio_dir`.
- `analysis`: `edge_seconds`, `silence_top_db`, `flow_profile` (`standard|deep-dj`), `resource_profile` (`default|background`), worker counts, `delete_audio_after_analysis`, `max_tracks`, `no_audio_analysis`, `refresh_cache`.
- `generation`: playlist sizing, `allow_reuse`, `genre_column|genre_clusters`, `beam_width`, `candidate_width`, `input_order_column`, `rate_transitions`, `transition_report_out`, `print_recommended_order`.
- `training`: `channels`/`videos` lists with labels, `output_path`, `max_videos`, search/metadata worker counts, `max_search_results`, label defaults.
- `exports.ytmusic` and `exports.youtube`: auth paths, titles, privacy.
- `logging.level`: `ERROR|WARNING|INFO|DEBUG`.

## Core Workflows
### Playlist analysis & generation
Run from CSV:
```powershell
python run.py --csv data/Playlist.csv
```
Run from YouTube playlist (fetch + optional audio analysis):
```powershell
python run.py --youtube-playlist <PLAYLIST_URL_OR_ID>
```
Key flags: `--playlist-size`, `--num-playlists`, `--allow-reuse`, `--beam-width`, `--candidate-width`, `--max-tracks`, `--edge-seconds`, `--silence-top-db`, `--flow-profile`, `--resource-profile`, `--download-workers`, `--analysis-workers`, `--refresh-cache/--no-refresh-cache`, `--audio-cache`, `--delete-audio-after-analysis/--keep-audio-cache`, `--rate-transitions`, `--transition-report-out`, `--print-recommended-order`, `--input-order-column`, `--create-ytmusic`, `--ytmusic-auth`, `--ytmusic-title`, `--ytmusic-privacy`, `--create-youtube`, `--youtube-client-secrets`, `--youtube-token`, `--youtube-title`, `--youtube-privacy`.
Outputs: `data/Playlist_reordered.csv` for full reorders, otherwise `data/Generated_playlists.csv`, plus optional transition report CSV.

### Scene → music (image + context matching)
Pick songs that fit a still frame (a movie scene, a photo, cover art). Mai reads
the image's colour and light — dominant palette, brightness, saturation, contrast,
warm/cool balance — and projects it into the same `valence / arousal / tension /
warmth` mood space the audio side uses, then ranks a track library by mood fit. A
free-text scene description adds narrative context (a red frame is a sunset romance
or a burning battlefield; the words disambiguate) and soft genre hints.
```powershell
python -m mai.scene_match --csv data/Library.csv --image scene.jpg `
  --scene-text "a tense chase through the city at night" --top-k 15 --order
```
Key flags: `--image`, `--scene-text` (either or both), `--text-weight` (blend of
caption vs colour mood, default 0.5), `--top-k`, `--order` (reorder the matches
into a flowing sequence via the transition engine), `--out` (write ranked CSV).
The colour path needs only `pillow`; set `MAI_CLIP_MODEL` (with `torch` +
`transformers`) to ground the image against mood/genre prompts in a learned CLIP
space, mirroring the audio side's CLAP seam.

### Scaling to large pools & evaluation
For a large catalogue, `mai.scene_index.SceneIndex` runs a staged retrieval funnel
so the expensive scene scorer only touches a few thousand survivors per query:
`hard filter (metadata) → ANN recall (mood space) → exact rerank → diversify
(MMR or k-DPP) → optional order`. Each stage has a fast path and an exact
numpy/pandas fallback: filtering uses Polars predicate pushdown when installed;
ANN recall auto-selects hnswlib → usearch → Faiss IVF-PQ (else an exact numpy
scan); `quantize_int8` gives a 4× coarse prefilter. The index persists as Parquet
+ npz (`save_parquet`, mmap-friendly) or a single pickle, supports incremental
`add`, and `query_batch` scores many scenes in one BLAS matmul for throughput.
```python
from mai.scene_index import build_scene_index
from mai.scene_match import build_scene_target
index = build_scene_index(library_df)                       # precompute once
target = build_scene_target(scene_text='a neon city chase at night')
hits = index.query(target, filters={'tempo': {'min': 110, 'max': 140}},
                   recall_k=2000, top_k=20, diversify=True, order=True)
```

The matcher is measurable, not just demoable. `mai.scene_eval` provides ranking
metrics (P@k, R@k, MRR, MAP, nDCG) and baseline retrievers (random, genre-lexicon,
CLAP seam) behind one `Retriever` protocol; `compare_retrievers` tabulates them on
a labelled benchmark. `mai.scene_dataset` defines the MAI-Bench JSONL schema, a
validator, a `make_synthetic_benchmark` smoke harness, and `ingest_cue_sheets` to
turn film/TV cue sheets (real scene↔track ground truth) into labelled examples.

For rigor, `scene_eval` also provides ablation flags on the Mai retriever
(`use_image` / `use_text` / `use_genre_boost`), bootstrap confidence intervals
(`bootstrap_ci`), and a paired bootstrap significance test (`paired_bootstrap_test`)
so a win over a baseline comes with a p-value, not just a higher number.

Three learned upgrades sit behind the always-on affect baseline: `mai.affect_probe`
distils any frozen embedding (CLAP/CLIP/MERT/descriptor) into the interpretable
`valence/arousal/tension/warmth` axes via a closed-form ridge probe (no torch,
weights are the explanation); `mai.cross_modal_model` trains a CLIP-style joint
image↔music space from cue-sheet pairs (symmetric InfoNCE, torch-guarded); and
`mai.scene_generation` closes the retrieval⊕generation loop — `scene_to_prompt`
turns a scene's affect into a MusicGen prompt so a fitting track can be *generated*
when none exists in the library (torch-guarded).

### Training scrape (positive transitions)
```powershell
python -m mai.training_scrape --config mai.toml
```
Use `--channel-url` to override configured sources, `--out` for the CSV path, `--max-videos`, `--max-search-results`, `--metadata-workers`, `--search-workers`, `--download-workers`, `--analysis-workers`, `--edge-seconds`, `--silence-top-db`, `--flow-profile`, `--resource-profile`, `--refresh-cache`, `--audio-cache`, `--cache-dir`. Produces a labeled transitions CSV (default `data/training/positive_transitions.csv`) and prints a scrape summary.

### Cache cleanup (optional)
Remove stale audio/temp/yt-dlp cache files:
```powershell
python -m mai.cache_cleanup --dry-run
python -m mai.cache_cleanup            # actually deletes
```

## Models and training
Mai blends hand-built transition components with two learned models. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design and the rationale behind each modelling choice, and [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) for the end-to-end pipeline.

Train both models from a scraped positive-transitions CSV:
```powershell
python -m mai.train --train-csv data/training/positive_transitions.csv
```
Outputs `data/cache/transition_model.joblib` (pairwise) and `data/cache/arc_model.joblib` (set-level arc). The transition model uses hard-negative mining (close-but-unchosen alternatives) and Camelot/tempo/mood interaction features, and prints a cross-validated AUC so you can judge data quality before generating. Key flags: `--negative-ratio`, `--hard-fraction`, `--model-backend {auto,forest,knn}`, `--device {cuda,cpu,auto}`, `--arc-profile {rise_peak_cool,rise,plateau,cool_down}`, `--skip-arc-model`.

Built for small datasets. With only a handful of scraped mixes the trainer leans on sample-efficient choices automatically: `--model-backend auto` picks a distance-weighted kNN when labelled positives are scarce (RandomForest once they are plentiful); the arc model learns from playlist *order* (real vs shuffled mixes) with a torch-free classifier, so no transition labelling is needed; and training prints a `recommended --transition-model-weight` that is shrunk toward 0 when AUC or data are weak, so the hand-built heuristics stay in charge until the learned model earns trust.

Use a trained model when generating:
```powershell
python run.py --csv data/Playlist.csv --transition-model-path data/cache/transition_model.joblib --transition-model-weight 0.35
```

Optional ML acceleration: install `torch` (GPU transition model + learned arc GRU) and, for cross-modal grounding, `transformers` with `MAI_AUDIO_ENCODER` / `MAI_CLAP_MODEL` set. Without them the descriptor representation, RandomForest scorer, and heuristic arc fit run as the always-on baseline.

Scrape precision is tunable via `MAI_RESOLUTION_MIN_SCORE`, `MAI_RESOLUTION_LOW_OVERLAP_MIN_OVERLAP`, and `MAI_RESOLUTION_LOW_OVERLAP_MIN_SCORE`.

## Caches and Outputs
- Playlist metadata cache: `data/cache/youtube_playlists/*.csv`.
- Audio feature cache: `data/cache/audio_features.sqlite` (tabular cache) plus optional `.csv` sibling; audio files in `data/audio_cache/` (pruned after analysis unless kept).
- Training caches under `data/cache/training/`: `channel_videos/*.json`, `video_metadata/*.json`, `search_results/*.json`, `source_tracks.sqlite`, `track_resolutions.sqlite`.
- Tools auto-downloaded on Windows when missing: `data/tools/deno/deno.exe` (JS runtime), `data/tools/ffmpeg/bin/` (ffmpeg/ffprobe).
- Outputs: reordered/generated playlists in `data/`, training transitions in `data/training/`.

## yt-dlp JS handling on Windows
- Default extractor opts skip YouTube player JS (`player_skip=js`) to avoid JS challenges.
- If JS is required, `mai` auto-detects runtimes and will download Deno to `data/tools/deno/deno.exe` on Windows. You can override the download URL with `MAI_DENO_WINDOWS_ZIP_URL=<zip_url>`.
- Node is skipped by default to avoid provider crashes; enable it with `MAI_YTDLP_ALLOW_NODE_RUNTIME=1` if you prefer your installed Node.
  - If offline, manually drop a working `deno.exe` into `data/tools/deno/` (or add `deno` to `PATH`) before rerunning.
  - If YouTube returns HTTP 429, try `MAI_YTDLP_MIN_REQUEST_INTERVAL=0.5` (seconds) or reduce `--metadata-workers` / `--search-workers`.
  - Optional backoff tuning: `MAI_YTDLP_RATE_LIMIT_RETRIES`, `MAI_YTDLP_RATE_LIMIT_BACKOFF`, `MAI_YTDLP_RATE_LIMIT_MAX_BACKOFF`.
  - Set `MAI_YTDLP_GLOBAL_PAUSE_ON_429=0` to disable pausing all workers when rate limits are detected.
- Optional: set `MAI_YTDLP_ENABLE_INTERNAL_CACHE=1` to keep yt-dlp’s own cache under `data/cache/yt_dlp_internal`.

## Progress UX
All CLIs use `CliProgressRenderer`, which emits heartbeat spinner updates every ~2s so long downloads, searches, or analyses don’t appear hung. Track-level notes surface when individual items advance.

## Future Vision
- Add local folder/audio-file ingestion alongside YouTube playlists, keeping the same transition engine.
- Ship evaluation reports that visualize weakest transitions and let users pin/ban tracks before export.
- Incremental training scrape: resume from the last scanned video per channel and append-only caches.
- Lightweight web UI for monitoring runs, viewing heartbeat progress, and downloading outputs.
- Per-user scoring tweaks: weights for energy/key/tempo/sentiment with presets saved in `mai.toml`.
- Export helpers for DJ tools (cue sheets, Rekordbox/Traktor-ready CSV with beat grids where available).
- Active-learning loop: surface low-confidence resolved tracks for quick human confirmation before training.
