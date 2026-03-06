# Mai

NLP + CV assisted playlist optimization for film scenes.

**Quickstart**
- `pip install -r requirements.txt`
- Add `.env` with `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET` for genre lookup.
- `python scripts/recommend_songs.py --scenes-json data/examples/scene_context.json --audio-dir <your_mp3_folder>`

**Project Layout**
- `src/mai/` core application code
- `main.py` CLI entrypoint (thin wrapper)
- `scripts/` thin CLI wrappers for utilities
- `configs/` default pipeline config
- `schemas/` JSON schemas for inputs/outputs
- `docs/` setup and improvements notes
- `notebooks/` experiments
- `data/` sample inputs and outputs

**Main Commands**
- `python scripts/extract_scene_text.py --input-text "scene context here"`
- `python scripts/extract_palette.py --input data/frames/scene_001 --window-size 30 --window-step 15`
- `python scripts/analyze_audio.py --audio-dir <your_mp3_folder> --out data/audio_features.json`
- `python scripts/recommend_songs.py --scenes-json data/examples/scene_context.json --audio-dir <your_mp3_folder> --out data/recommendations.json`

**Input Notes**
- Scene JSON follows `schemas/scene_context.schema.json`.
- `frames_path` per scene is optional but recommended.
- For better Spotify matching, provide metadata CSV with `path,title,artist` via `--metadata-csv`.
 - Optional audio tag genre mapping lives at `configs/genre_map.json`.

**Embeddings Notes**
- Audio embeddings use `musicnn` + `tensorflow`. If install fails on your Python version, run with `--no-embeddings` or switch to Python 3.10/3.11.
 - If `musicnn` fails to install due to old numpy pins, you can still run everything with `--no-embeddings` and rely on DSP + Spotify genre tags.
