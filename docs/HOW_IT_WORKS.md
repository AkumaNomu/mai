# How Mai Works

This document explains the current Mai pipeline end to end: input loading, local audio analysis, genre resolution, transition scoring, playlist generation, and export.

## High-level flow

Mai runs in this order:

1. Load a song pool from CSV or a public YouTube playlist.
2. Optionally download and analyze audio locally.
3. Normalize columns and derive sentiment features.
4. Resolve genre families and style-cluster fallback groups.
5. Build a directed transition score from each song's outro into each candidate next song's intro.
6. Generate one reordered playlist or multiple optimized playlists from the pool.
7. Write CSV output.
8. Optionally export the results to YouTube Music, standard YouTube, or both.

The CLI entry point is `run.py`.

## Project config

Mai reads project defaults from `mai.toml`.

Precedence is:
1. CLI flags
2. `mai.toml`
3. built-in defaults in `mai/config.py`

Both `run.py` and `python -m mai.training_scrape` support:
- `--config <PATH>`
- `--no-config`

The tracked config stores paths and behavior defaults, not secret contents.

For training sources, `mai.toml` uses repeatable `[[training.channels]]` blocks.
The balanced scraper defaults are:
- `training.max_search_results = 5`
- `training.metadata_workers = 4`
- `training.search_workers = 4`

## Entry point and orchestration

`run.py` is responsible for:

1. parsing CLI flags
2. loading metadata from `--csv` or `--youtube-playlist`
3. running audio analysis unless `--no-audio-analysis` is set
4. normalizing numeric/audio columns
5. computing directed transition scores
6. generating one or more playlists
7. writing `data/Playlist_reordered.csv` or `data/Generated_playlists.csv`
8. exporting playlists when requested

Important flags:
- `--playlist-size`
- `--num-playlists`
- `--allow-reuse`
- `--cache-dir`
- `--refresh-cache`
- `--genre-column`
- `--genre-clusters`
- `--beam-width`
- `--candidate-width`
- `--edge-seconds`
- `--silence-top-db`
- `--flow-profile`
- `--create-ytmusic`
- `--create-youtube`

## Data loading

Helpers live in `mai/data.py` and `mai/youtube_integration.py`.

### CSV input

`load_csv_playlist()` reads a CSV into a dataframe and strips column whitespace.

### YouTube playlist metadata input

`fetch_youtube_playlist_tracks()` uses `yt-dlp` flat extraction to pull metadata without downloading audio first.

By default it also caches the playlist metadata in `data/cache/youtube_playlists` and reuses it on later runs unless `--refresh-cache` is set.

Mai keeps fields such as:
- `title`
- `artist`
- `video_id`
- `duration`
- `duration_seconds`
- `url`
- `uploader`
- `channel`
- `channel_id`
- `description`
- `tags`
- `category`

### Column normalization

Before scoring, `run.py` calls:
- `normalize_audio_feature_columns()`
- `ensure_audio_columns()`
- `add_log_tempo()`

Numeric coercion covers:
- full-track audio features
- `intro_*` and `outro_*` features
- `sentiment_*`
- `genre_confidence`

## Local audio analysis

Audio analysis lives in `mai/audio_analysis.py`.

This is mainly used for YouTube inputs because YouTube does not expose the descriptors Mai needs.

### Download and conversion

`download_youtube_audio()`:

1. creates the cache directory
2. prefers cached `<video_id>.wav`
3. deletes stale non-WAV cache entries for the same video
4. ensures `ffmpeg` and `ffprobe` exist
5. uses `yt-dlp` to download audio and convert it to WAV

The WAV cache lives in `data/audio_cache`.

The download path is defensive:
- forced IPv4
- larger socket timeout
- `yt-dlp` retries
- outer retry loop with backoff

On Windows, Mai can auto-download `ffmpeg` and `ffprobe` into `data/tools/ffmpeg`.

### Non-silent edge windows

Mai does not analyze the raw first and last 30 seconds.

Instead it:

1. splits the waveform into non-silent intervals with `librosa.effects.split()`
2. gathers non-silent audio from the beginning until the requested edge window is filled
3. gathers non-silent audio from the end until the requested edge window is filled

This keeps transition analysis focused on meaningful audio instead of dead air.

Bookkeeping columns:
- `intro_seconds_used`
- `outro_seconds_used`
- `intro_leading_silence_s`
- `outro_trailing_silence_s`

Mai also caches extracted feature rows globally in `data/cache/audio_features.sqlite`.

Each cache row stores:
- `video_id`
- compact `analysis_signature` for the analysis settings
- derived `sentiment_*` columns alongside the audio features

Mai loads that SQLite-backed table once per run and indexes it in memory by `video_id + analysis_signature` for fast cache lookup while analyzing a playlist.
If a row exists for the same `video_id` with matching settings, Mai reuses it instead of decoding the audio again.
If the settings differ, Mai can keep a separate row for that alternate signature instead of bloating every row with repeated config metadata.

Legacy per-video JSON cache files under `data/cache/audio_features/` are treated as a migration source and imported into SQLite when needed.

### Base song, intro, and outro features

`_compute_features()` extracts:
- `tempo`
- `key`
- `mode`
- `loudness`
- `danceability`
- `energy`
- `speechiness`
- `acousticness`
- `liveness`
- `valence`
- `rms`
- `spectral_centroid`
- `spectral_bandwidth`
- `spectral_flatness`
- `spectral_rolloff`
- `zcr`
- `onset_strength`
- `harmonic_ratio`
- `mfcc1` to `mfcc5`

Mai computes these for:
- the full song
- `intro_*`
- `outro_*`

### Deep edge-flow features

With `--flow-profile deep-dj`, `analyze_audio_file()` also derives DJ-style handoff features.

Intro-side:
- `intro_attack_time_s`
- `intro_rise_slope`
- `intro_onset_density`
- `intro_flux_peak`
- `intro_beat_stability`
- `intro_pad_silence_s`
- `intro_downbeat_strength`
- `intro_chroma_stability`

Outro-side:
- `outro_release_time_s`
- `outro_decay_slope`
- `outro_abruptness`
- `outro_onset_density`
- `outro_flux_peak`
- `outro_beat_stability`
- `outro_tail_silence_s`
- `outro_downbeat_strength`
- `outro_chroma_stability`

These are meant to capture:
- abrupt cuts versus smooth fades
- quick attacks versus slow bloom intros
- rhythmic density near the transition edge
- beat-grid stability
- silence padding at the handoff
- harmonic stability at the edge

## Sentiment derivation

Sentiment logic lives in `mai/sentiment.py`.

Mai does not currently analyze lyrics. It derives compact audio sentiment from existing acoustic features.

For the full track, intro, and outro it writes:
- `sentiment_valence`
- `sentiment_arousal`
- `sentiment_tension`
- `sentiment_warmth`

These are heuristic combinations of:
- energy
- danceability
- valence
- acousticness
- speechiness
- harmonic ratio
- onset activity
- brightness
- roughness proxies

## Genre resolution

Genre resolution lives in `mai/genre.py`.

The goal is to produce stable genre-family labels when Mai has enough evidence, while still supporting large-pool balancing when it does not.

### Resolution order

Mai resolves in this order:

1. explicit input genre columns
   - `--genre-column`
   - `genre`
   - `genres`
2. local metadata parsing
   - title
   - artist
   - uploader
   - channel
   - tags
   - category
   - description
3. local audio heuristic classification
4. `style_cluster` fallback

### Output columns

Mai always writes:
- `genre_primary`
- `genre_confidence`
- `genre_source`
- `style_cluster`
- `mix_group`

`mix_group` is what the generator actually balances:
- if `genre_confidence >= 0.55`, it uses `genre_primary`
- otherwise it uses `style_cluster`

### Canonical genre families

The resolver normalizes into:
- `pop`
- `rock`
- `indie_alt`
- `electronic`
- `house_techno`
- `hip_hop_rap`
- `rnb_soul`
- `jazz`
- `ambient_chill`
- `folk_acoustic`
- `latin_world`
- `classical_score`
- `metal_punk`
- `unknown`

### Style cluster fallback

`style_cluster` is inferred from audio + sentiment features with `KMeans`.

This is not a true musicological genre label. It is a fallback diversity bucket that keeps generation working when real genre evidence is weak.

## Tonal compatibility

Tonal logic lives in `mai/tonal.py`.

Mai uses Krumhansl-Kessler profiles to compare:
- `outro_key`, `outro_mode`
- against `intro_key`, `intro_mode`

This gives a directed tonal score instead of a simple whole-track key equality check.

## Directed transition scoring

Transition scoring lives in `mai/playlist_generation.py`.

The matrix is directed: `A -> B` is scored from how song A ends and how song B begins.

### Score components

`compute_transition_scores()` combines up to six components:

1. `edge_flow_score`
2. `structure_cadence_score`
3. `timbre_score`
4. `groove_score`
5. `sentiment_score`
6. `tonal_score`

Default weights:
- `edge_flow_score`: `0.30`
- `structure_cadence_score`: `0.20`
- `timbre_score`: `0.20`
- `groove_score`: `0.10`
- `sentiment_score`: `0.10`
- `tonal_score`: `0.10`

If a component cannot be built because the required columns are missing, Mai skips it and renormalizes over the remaining components.

### Edge-flow scoring

`edge_flow_score` rewards complementary handoffs such as:
- abrupt outro -> fast hard intro
- long release -> gradual intro rise
- tail silence -> harder next entrance is allowed
- strong downbeat exit -> strong downbeat entry
- stable harmonic tail -> stable harmonic intro

It also applies a penalty when a hard intro follows an ending that does not leave enough space for it.

### Structure and cadence scoring

`structure_cadence_score` compares boundary shape rather than pure timbral similarity. It looks at:
- onset density
- transient peak intensity
- beat stability
- silence handoff
- downbeat strength
- chroma stability

### Timbre, groove, sentiment, tonal

The other components are more similarity-like:
- timbre uses MFCCs and spectral descriptors
- groove uses tempo, energy, loudness, onset strength, and danceability
- sentiment uses intro/outro sentiment vectors
- tonal uses Krumhansl-Kessler transition similarity

## Playlist generation

The generator also lives in `mai/playlist_generation.py`.

Mai uses beam search instead of a simple greedy walk so it can recover from locally tempting but globally bad decisions.

### Seed selection

Seeds are chosen from songs with strong outgoing transition quality, plus bonuses for underused mix groups and confident canonical genres.

### Per-step path scoring

When extending a playlist, Mai starts from the transition score and then applies soft path constraints:
- heavier penalty for artist adjacency than genre adjacency
- penalty when one mix group grows beyond about 45 percent of the playlist
- reward for adding a new mix group
- reward for bringing in new confident genre families early
- reward for cross-genre jumps only when the transition score is already strong
- penalty for weak cross-genre jumps

These are soft preferences, not hard rules, so a very strong transition can still override them.

### Multiple playlists

`generate_playlist_paths()` can build several playlists from the same pool:
- by default songs are used once across playlists
- `--allow-reuse` lets songs appear in more than one output playlist

`playlists_to_dataframe()` then writes:
- `playlist_index`
- `position`
- `playlist_name`
- `transition_score_from_previous`
- `transition_rating_from_previous`

If the dataframe already contains resolved genre columns from a previous run, Mai reuses them instead of reclustering and reclassifying again.

## Output CSVs

Normal output files are:
- `data/Playlist_reordered.csv`
- `data/Generated_playlists.csv`

The CSVs include:
- source metadata
- full-song audio features
- intro/outro features
- sentiment features
- deep edge-flow features when available
- genre resolution columns
- playlist ordering columns

`transition_rating_from_previous` maps the score from the previous song into:
- `excellent`
- `strong`
- `good`
- `mixed`
- `rough`

The first row in each playlist is labeled `start`.

## Rating an existing order

`run.py` can score the current playlist order before showing the recommended one.

Useful flags:
- `--rate-transitions`
- `--input-order-column <COLUMN_NAME>`
- `--print-recommended-order`

When `--rate-transitions` is enabled, Mai writes a detailed CSV comparing:
- `input_order`
- `recommended_order`

For each adjacent pair it records:
- source and destination labels
- numeric transition score
- rating band
- rank of the chosen next song among the remaining candidates
- best available next-song alternative and score gap

## Positive transition scraping

Mai also has a training-data scraper for building positive transition labels from expert channel uploads such as `@mai_dq`.

`python -m mai.training_scrape`:
- reads one or more `[[training.channels]]` entries from `mai.toml`
- enumerates videos from each channel's videos tab
- caches the raw channel enumeration under `data/cache/training/channel_videos`
- fetches and caches each source video's full metadata payload under `data/cache/training/video_metadata`
- builds source-track rows from description timestamps first
- falls back to watch-page chapter/music metadata when descriptions are missing or insufficient
- caches normalized parsed source-track rows in `data/cache/training/source_tracks.sqlite`
- searches YouTube for the listed songs and caches raw search payloads under `data/cache/training/search_results`
- caches per-track resolution outcomes in `data/cache/training/track_resolutions.sqlite`
- runs the same audio analysis pipeline used for normal playlist inputs
- writes one `excellent` row per adjacent song handoff in the original mix order

Balanced scraping speedups include:
- a bounded worker pool for source-video metadata fetches
- a bounded worker pool for track searches
- in-run deduplication of repeated normalized search queries
- batched writes to `source_tracks.sqlite` and `track_resolutions.sqlite`

`training.max_search_results` is the per-track YouTube candidate cap used during resolution.
The default `5` keeps search broad enough to avoid many weak first hits without turning each row into an expensive candidate sweep.

Description timestamps are the authoritative timeline when they produce a usable tracklist.
Chapter/music metadata is used as fallback and enrichment, and Mai records whether each source row came from:
- `description`
- `chapters`
- `description+chapters`

Mai never bridges across unresolved tracks.
If a mix contains `A -> B -> C` and `B` cannot be resolved or analyzed:
- Mai does not emit `A -> B`
- Mai does not emit `B -> C`
- Mai does not synthesize a fake `A -> C`

The broken adjacency is still preserved in the cached source-track and resolution tables so reruns and audits can see the gap.

The final training CSV is aggregated across configured channels, sorted by source mix `video_id`, uses `video_id` as the first column, preserves raw scraped track strings and provenance, and includes resolved song ids plus prefixed `from_` / `to_` analysis columns for model training.
Pairs are only built within one source mix `video_id`, so channels and playlists never bleed into each other.

## Export to YouTube Music

YouTube Music export lives in `mai/ytmusic_integration.py`.

`run.py` loads `ytmusicapi` only when `--create-ytmusic` is requested.

Mai:
1. creates a playlist
2. adds video ids in batches
3. creates one destination playlist per generated output playlist

If there are multiple generated playlists, titles are suffixed with `01`, `02`, and so on.

## Export to standard YouTube

Standard YouTube export lives in `mai/youtube_export.py`.

It uses the YouTube Data API with desktop OAuth:
- client secrets default: `data/youtube_client_secret.json`
- cached token default: `data/youtube_token.json`
- loopback callback host: `127.0.0.1`
- export title default: `auto`

If the browser cannot complete the loopback callback, the exporter falls back to asking for the final redirected `http://127.0.0.1:...` URL so the OAuth token exchange can still finish.

`run.py` loads the Google API stack only when `--create-youtube` is requested.

The exporter:
1. authenticates with OAuth
2. creates the playlist
3. inserts videos in order
4. retries API requests
5. logs and skips individual failed insertions instead of aborting the whole export

When the source input came from `--youtube-playlist`, `auto` resolves to `<playlist name> mai enhanced`.
If there are multiple generated playlists, Mai creates one YouTube playlist per output playlist and suffixes the titles with `01`, `02`, and so on.
Any explicit non-`auto` title overrides that behavior.

## Progress logging

Mai reports progress through Python logging.

Useful levels:
- `ERROR`
- `WARNING`
- `INFO`
- `DEBUG`

Progress messages include:
- playlist metadata fetch
- audio analysis per track
- cache reuse for metadata and extracted features
- transition component construction
- genre-resolution summary
- playlist generation progress
- YouTube export progress

## Current limitations

- Sentiment is still audio-derived only.
- Genre resolution is heuristic and mostly local by design.
- `style_cluster` is a fallback grouping tool, not a canonical genre label.
- Beam search is practical and tunable, but still heuristic rather than globally optimal.
- Standard YouTube export needs a real Google OAuth client setup to run live.

## Good next extensions

Natural next steps from here:
- add lyric sentiment and blend it with audio sentiment
- learn transition weights from labeled good and bad handoffs
- add target playlist arcs such as rise, plateau, cool-down
- preview crossfades directly from the waveform
- add a small UI for approving, editing, and exporting generated playlists
