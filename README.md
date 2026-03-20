# Mai - transition-aware playlist generation

Mai reorders playlists and generates new ones by scoring how one song ends against how the next song begins.

It can:
- fetch tracks from a public YouTube playlist
- download audio locally and analyze it
- derive intro and outro features from non-silent 30 second edge windows
- infer audio sentiment and genre families
- generate one reordered playlist or multiple `n`-sized playlists from a larger pool
- export the result to YouTube Music and standard YouTube playlists

Deep dive

- See `docs/HOW_IT_WORKS.md` for the full pipeline and scoring details

## Quick start

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
& .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

2. Reorder a CSV using the full pool:

```powershell
python run.py --csv data/Playlist.csv
```

This writes `data/Playlist_reordered.csv`.

3. Generate several playlists from a larger pool:

```powershell
python run.py --csv data/Playlist.csv --playlist-size 25 --num-playlists 4
```

This writes `data/Generated_playlists.csv`.

## Config

Mai now reads project defaults from `mai.toml`.

- CLI flags override `mai.toml`
- `mai.toml` overrides built-in defaults
- `--config <PATH>` loads a different config file
- `--no-config` ignores TOML and uses only CLI plus built-in defaults

Keep secrets out of `mai.toml`. Store file paths there instead, such as:
- `data/youtube_client_secret.json`
- `data/youtube_token.json`
- `data/ytmusic_auth.json`

Training sources are configured by repeating `[[training.channels]]` blocks in `mai.toml`.
The default audio-analysis parallelism is:
- `analysis.download_workers = 4`
- `analysis.analysis_workers = 4`
- `analysis.resource_profile = "default"` or `"background"` for gentler background analysis

The balanced scraper defaults are:
- `training.max_search_results = 5`
- `training.metadata_workers = 4`
- `training.search_workers = 4`

## YouTube input

Fetch a public YouTube playlist and analyze audio locally:

```powershell
python run.py --youtube-playlist <PLAYLIST_URL_OR_ID>
```

Useful analysis flags:
- `--cache-dir data/cache`
- `--refresh-cache`
- `--edge-seconds 30`
- `--silence-top-db 35`
- `--flow-profile deep-dj`
- `--download-workers 4`
- `--analysis-workers 4`
- `--log-level INFO`
- `--log-level DEBUG`

Skip audio analysis if the input already contains the needed columns:

```powershell
python run.py --youtube-playlist <PLAYLIST_URL_OR_ID> --no-audio-analysis
```

Audio analysis uses `yt-dlp` + `librosa`. On Windows, Mai will try to auto-download `ffmpeg` and `ffprobe` into `data/tools/ffmpeg` if they are missing.
When there are uncached tracks, Mai now parallelizes YouTube audio downloads and local feature extraction while keeping cache writes centralized.
Use `--resource-profile background` when you want audio analysis to back off and run more gently in the background.

Mai now reuses cached data by default:
- playlist metadata cache in `data/cache/youtube_playlists`
- extracted feature cache in `data/cache/audio_features.sqlite`
- downloaded audio cache in `data/audio_cache`

Use `--refresh-cache` when you want to ignore those caches and rebuild from scratch.
To clean redundant cache files that are safe to regenerate, run:

```powershell
python -m mai.cache_cleanup --dry-run
python -m mai.cache_cleanup
```

To analyze audio files that are already sitting in `data/audio_cache` and delete each file right after its features are saved into the global SQLite cache, run:

```powershell
python -m mai.audio_analysis
```

The audio feature cache is global across playlists, keyed by `video_id`, and stores the analysis settings used for that row.
If the same video is reanalyzed with different settings, Mai overwrites that row with the latest feature payload.
It also acts as Mai's unified song-memory registry by keeping reusable metadata text, normalized `context_text`, and derived sentiment columns alongside the audio features.

## What gets analyzed

Mai keeps the whole-song features, but it also focuses on the first and last non-silent audio around the transition boundary.

Base edge descriptors include:
- `intro_*`
- `outro_*`
- `intro_seconds_used`
- `outro_seconds_used`
- `intro_leading_silence_s`
- `outro_trailing_silence_s`

The default `deep-dj` flow profile adds stronger transition-specific features:
- `intro_attack_time_s`
- `intro_rise_slope`
- `intro_onset_density`
- `intro_flux_peak`
- `intro_beat_stability`
- `intro_pad_silence_s`
- `intro_downbeat_strength`
- `intro_chroma_stability`
- `outro_release_time_s`
- `outro_decay_slope`
- `outro_abruptness`
- `outro_onset_density`
- `outro_flux_peak`
- `outro_beat_stability`
- `outro_tail_silence_s`
- `outro_downbeat_strength`
- `outro_chroma_stability`

These are used to catch things like:
- abrupt endings into fast intros
- long fade-outs into gradual risers
- silence at the handoff allowing a harder next entrance
- strong beat exits into strong beat entries

## Genre resolution

Mai resolves genre information in this order:

1. `--genre-column`, or existing `genre` / `genres`
2. local metadata parsing from fields like title, artist, uploader, channel, tags, category, and description
3. a local audio heuristic classifier over audio + sentiment features
4. `style_cluster` fallback when confidence is low

It writes:
- `genre_primary`
- `genre_confidence`
- `genre_source`
- `style_cluster`
- `mix_group`

`mix_group` is what generation uses for balancing:
- confident canonical genre when `genre_confidence >= 0.55`
- otherwise inferred `style_cluster`

Canonical genre families:
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

## Generation

Generate multiple optimized playlists from a large pool:

```powershell
python run.py --csv data/Playlist.csv --playlist-size 20 --num-playlists 5
```

Useful generation flags:
- `--genre-column genre`
- `--genre-clusters 10`
- `--beam-width 8`
- `--candidate-width 25`
- `--allow-reuse`

The transition score combines:
- `edge_flow_score`
- `structure_cadence_score`
- `timbre_score`
- `groove_score`
- `sentiment_score`
- `tonal_score`

Generated playlist CSVs now include:
- `transition_score_from_previous`
- `transition_rating_from_previous`

Transition ratings use five bands:
- `excellent`
- `strong`
- `good`
- `mixed`
- `rough`

Generation then adds soft path constraints so playlists:
- avoid artist back-to-back repeats
- avoid one genre dominating more than about 45 percent
- prefer controlled cross-genre jumps only when the transition score is already strong
- reward bringing in new confident genre families early

Rate the current playlist order and write a detailed transition report:

```powershell
python run.py --csv data/Playlist.csv --rate-transitions
```

If your CSV already has an order column, point Mai at it:

```powershell
python run.py --csv data/Playlist.csv --input-order-column track_number --rate-transitions
```

Print the recommended order directly in the terminal:

```powershell
python run.py --csv data/Playlist.csv --print-recommended-order
```

## Training data

Build positive transition training rows from one or more configured channels:

```powershell
python -m mai.training_scrape
```

Useful flags:
- `--channel-url https://www.youtube.com/@mai_dq/videos`
- `--out data/training/positive_transitions.csv`
- `--max-videos 10`
- `--max-search-results 5`
- `--metadata-workers 4`
- `--search-workers 4`
- `--download-workers 4`
- `--analysis-workers 4`
- `--config alt_mai.toml`
- `--no-config`

Add more source channels by repeating the block in `mai.toml`:

```toml
[[training.channels]]
name = "mai_dq"
url = "https://www.youtube.com/@mai_dq/videos"
label = "excellent"
label_source = "mai_dq_mix_curation"

[[training.channels]]
name = "another_channel"
url = "https://www.youtube.com/@another_channel/videos"
label = "excellent"
label_source = "another_channel_mix_curation"
```

The scraper:
- reads source channels from `[[training.channels]]` in `mai.toml`
- caches every network-derived artifact it touches under `data/cache/training`
- reads timestamped tracklists from descriptions first
- falls back to YouTube watch-page chapter/music metadata when descriptions are missing or incomplete
- searches YouTube for the listed songs
- runs Mai's normal audio analysis on the resolved song videos
- writes one `excellent` row per adjacent transition in the original expert order

`training.max_search_results` is the per-track YouTube candidate cap used during track resolution.
The default `5` is the balanced setting: enough room to avoid many bad first hits without turning every search into a long candidate scan.

Training caches now include:
- raw channel video enumeration
- raw per-video metadata payloads
- normalized parsed source tracks
- raw search results, including failed lookups
- per-track resolution outcomes such as `resolved`, `unavailable`, `no_match`, and `analysis_failed`

If `yt-dlp` hits age-gated or login-only videos, export a YouTube cookies file to the project root as `cookies.txt`.
Mai now passes that file automatically to `yt-dlp` for channel scraping, per-video metadata fetches, playlist reads, and audio downloads.
If `yt-dlp` warns that the provided YouTube cookies are no longer valid, re-export `cookies.txt` from the browser before rerunning age-gated or account-required videos.

Missing tracks never bridge the gap. If a mix goes `A -> B -> C` and `B` is unavailable, Mai drops the broken pairs instead of inventing a false `A -> C` positive transition. The gap is still preserved in the training caches for reruns and auditing.

The output CSV is aggregated across configured channels, sorted by source mix `video_id`, uses `video_id` as the first column, and includes source provenance plus prefixed `from_` / `to_` analysis features for training.
Positive pairs are only built within one source mix `video_id`, so playlists and channels never get cross-wired in the training output.

## Export to YouTube Music

1. Create a `ytmusicapi` auth file:

```powershell
ytmusicapi browser --file data/ytmusic_auth.json
```

2. Export:

```powershell
python run.py --youtube-playlist <PLAYLIST_URL_OR_ID> --create-ytmusic --ytmusic-auth data/ytmusic_auth.json
```

Useful flags:
- `--ytmusic-title "Mai Mix"`
- `--ytmusic-privacy PRIVATE`

If multiple playlists are generated, Mai creates one YouTube Music playlist per `playlist_index`, suffixing titles with `01`, `02`, and so on.

## Export to standard YouTube

1. Create a Google Cloud desktop OAuth client for the YouTube Data API.
2. Save the client secrets JSON to `data/youtube_client_secret.json`.
3. Run:

```powershell
python run.py --youtube-playlist <PLAYLIST_URL_OR_ID> --create-youtube
```

On first run, Mai opens the desktop OAuth flow on `127.0.0.1` and caches the token in `data/youtube_token.json`.
If the browser cannot finish the local callback, Mai falls back to prompting for the final redirected `http://127.0.0.1:...` URL so the token exchange can still complete.

Useful flags:
- `--youtube-client-secrets data/youtube_client_secret.json`
- `--youtube-token data/youtube_token.json`
- `--youtube-title auto`
- `--youtube-title "Mai Mix"`
- `--youtube-privacy unlisted`

When the input came from `--youtube-playlist`, standard YouTube export now uses `<playlist name> mai enhanced` by default.
If multiple playlists are generated, Mai suffixes that base title with `01`, `02`, and so on.
Set `--youtube-title` to any non-`auto` value to override that behavior.

If multiple playlists are generated, Mai creates one standard YouTube playlist per `playlist_index`, also suffixing titles with `01`, `02`, and so on.

Item insertion is retry-aware and continues past individual failures, logging skipped video ids instead of aborting the whole export.

## Progress output

Use:

```powershell
python run.py --youtube-playlist <PLAYLIST_URL_OR_ID> --log-level INFO
python run.py --youtube-playlist <PLAYLIST_URL_OR_ID> --log-level DEBUG
```

Mai reports:
- metadata fetch progress
- per-track audio analysis progress
- transition scoring progress
- playlist generation progress
- cache reuse for metadata and audio features
- transition component construction
- playlist export progress

The CLI also shows lightweight progress bars during transition scoring, playlist generation, and training scraping.

## Performance tips

Fastest wins:
- let Mai reuse the default caches
- keep `--flow-profile deep-dj` only when you want the richer edge model
- use `--max-tracks` while iterating on scoring changes
- keep `--candidate-width` and `--beam-width` modest when testing

On reruns, Mai also reuses:
- existing audio features already present in the dataframe
- existing resolved genre columns when they are already in the CSV

## Repository layout

- `mai/` - audio analysis, genre resolution, transition scoring, and export helpers
- `run.py` - CLI entry point
- `docs/HOW_IT_WORKS.md` - architecture walkthrough
- `requirements.txt` - dependencies

## Notes

- Sentiment is currently audio-derived, not lyric-derived.
- Standard YouTube export is optional and only loads its Google API dependencies when `--create-youtube` is used.
- YouTube Music export is also optional and only loads `ytmusicapi` when `--create-ytmusic` is used.
