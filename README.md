# Mai
Transition-aware playlist tools for YouTube: reorder existing playlists, generate new ones, and scrape curated transitions for training data.

## Current Capabilities
- Pull a public YouTube playlist (metadata only) or read a CSV, normalize columns, and enrich with audio sentiment, key, tempo, and edge-intro/outro features.
- Score directed transitions and generate either a single reordered playlist or multiple fixed-size playlists with beam search, genre balancing, and optional reuse controls.
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
