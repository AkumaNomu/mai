# Mai Setup Guide

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Configure Spotify Credentials (for genre lookup)

Create `.env` in project root:

```bash
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
```

Without these values, the pipeline still runs but skips Spotify genre enrichment.

## 3. Prepare Inputs

- Candidate songs in a folder with local audio files (`.mp3`, `.wav`, `.flac`, `.m4a`, `.ogg`, `.aac`)
- Scene context JSON (`schemas/scene_context.schema.json`)
- Optional frame folders per scene via `frames_path`
- Optional metadata CSV with columns `path,title,artist` for better Spotify matching

Example scene JSON:

```json
{
  "scenes": [
    {
      "id": "scene_001",
      "context": "Main character boxing in gym, punching bags in frustration after his brother was killed by the mafia.",
      "frames_path": "data/frames/scene_001"
    }
  ]
}
```

## 4. Run Pipeline

```bash
python scripts/recommend_songs.py --scenes-json data/examples/scene_context.json --audio-dir <your_mp3_folder> --out data/recommendations.json
```

## 5. Useful Commands

```bash
python main.py recommend --scenes-json data/examples/scene_context.json --audio-dir <your_mp3_folder>
python main.py analyze-audio --audio-dir <your_mp3_folder> --out data/audio_features.json
python main.py extract-scene --input-text "hero trains in silence before final fight"
python main.py extract-palette --input data/frames/scene_001 --window-size 30 --window-step 15
```

## 6. Embeddings and Audio Genre

- Embeddings use `musicnn` + `tensorflow`.
- If embeddings fail to install on your Python version, run with `--no-embeddings` or use Python 3.10/3.11.
- Audio tag to genre mapping is in `configs/genre_map.json`.
 - If `musicnn` fails due to numpy pinning, the pipeline still works without embeddings.
