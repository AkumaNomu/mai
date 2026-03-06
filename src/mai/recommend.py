import argparse
import json
from pathlib import Path

import numpy as np

from .audio import analyze_library, clamp01
from .cv import extract_palette
from .nlp import compute_scene_features, tokenize


DEFAULT_CONFIG = {
    "cv": {
        "k": 5,
        "resize": 200,
        "max_images": 0,
        "stride": 1,
        "max_pixels": 5000,
        "sample_total": 200000,
        "window_size": 30,
        "window_step": 15,
        "fps": 1.0,
        "seed": 7,
    },
    "audio": {
        "sample_seconds": 90.0,
        "sample_offset": 0.0,
        "sr": 22050,
        "spotify_market": "US",
        "segment_count": 3,
        "segment_strategy": "equal",
        "embedding_backend": "musicnn",
        "embedding_dim": 200,
        "cache_version": 2,
    },
    "matching": {
        "weights": {
            "audio": 0.6,
            "genre": 0.2,
            "genre_audio": 0.1,
            "embedding": 0.1,
        },
        "feature_weights": {
            "valence": 0.15,
            "arousal": 0.15,
            "tension": 0.15,
            "energy": 0.1,
            "tempo": 0.1,
            "brightness": 0.1,
            "loudness": 0.08,
            "dynamic_range": 0.07,
            "hpr": 0.06,
            "rhythm_complexity": 0.07,
            "timbral": 0.07,
        },
    },
    "genre_rules": {
        "emotion": {
            "joy": ["pop", "dance pop", "funk", "disco", "indie pop"],
            "sadness": ["ambient", "acoustic", "piano", "singer-songwriter", "sad"],
            "anger": ["metal", "hard rock", "industrial", "aggressive rap", "punk"],
            "fear": ["dark ambient", "horror", "soundtrack", "industrial", "drone"],
            "disgust": ["industrial", "noise", "experimental"],
            "surprise": ["cinematic", "orchestral", "soundtrack", "electronic"],
            "trust": ["neo soul", "r&b", "acoustic", "lo-fi", "folk"],
            "anticipation": ["hip hop", "trap", "electronic", "drum and bass", "cinematic"],
        },
        "mood": {
            "romance": ["r&b", "soul", "jazz", "dream pop"],
            "awe": ["orchestral", "ambient", "post-rock", "cinematic"],
            "whimsy": ["indie pop", "folk", "chamber pop"],
            "noir": ["dark jazz", "jazz", "trip hop", "blues"],
            "grief": ["ambient", "neo-classical", "acoustic"],
            "anxiety": ["minimal techno", "dark ambient", "industrial"],
            "resolve": ["hip hop", "rock", "cinematic"],
            "violence": ["metal", "hardcore", "industrial"],
            "crime": ["trap", "drill", "dark hip hop"],
        },
        "setting": {
            "urban": ["hip hop", "trap", "r&b"],
            "night": ["synthwave", "dark pop", "trip hop"],
            "nature": ["ambient", "neo-classical", "post-rock", "world"],
            "weather_rain": ["jazz", "ambient", "lo-fi"],
            "weather_snow": ["piano", "ambient", "neo-classical"],
        },
    },
}


def load_config(path: Path | None) -> dict:
    if not path or not path.exists():
        return DEFAULT_CONFIG
    loaded = json.loads(path.read_text(encoding="utf-8"))
    merged = DEFAULT_CONFIG.copy()
    for key, value in loaded.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def load_scene_contexts(scene_json: Path) -> list[dict]:
    obj = json.loads(scene_json.read_text(encoding="utf-8"))
    scenes = obj.get("scenes", [])
    if not isinstance(scenes, list):
        raise ValueError("Scene JSON must contain a list field named 'scenes'")
    out: list[dict] = []
    for idx, scene in enumerate(scenes):
        context = str(scene.get("context", "")).strip()
        if not context:
            continue
        out.append(
            {
                "id": str(scene.get("id", f"scene_{idx:03d}")),
                "context": context,
                "frames_path": str(scene.get("frames_path", "")).strip(),
            }
        )
    return out


def aggregate_cv_metrics(cv_result: dict) -> dict:
    if "metrics" in cv_result:
        return cv_result["metrics"]
    windows = cv_result.get("windows", [])
    if not windows:
        return {}

    numeric_fields = [
        "contrast",
        "brightness",
        "saturation",
        "vibrance",
        "colorfulness",
        "temperature",
    ]
    agg = {}
    for field in numeric_fields:
        vals = [w.get("metrics", {}).get(field) for w in windows]
        vals = [v for v in vals if isinstance(v, (int, float))]
        agg[field] = float(sum(vals) / len(vals)) if vals else 0.0
    return agg


def _emotion_target(nlp_features: dict, cv_metrics: dict) -> dict:
    nlp_valence = float(nlp_features.get("valence", 0.0))
    nlp_arousal = float(nlp_features.get("arousal", 0.0))
    nlp_tension = float(nlp_features.get("tension", 0.0))

    temperature = float(cv_metrics.get("temperature", 0.0))
    contrast = float(cv_metrics.get("contrast", 0.0))
    brightness = float(cv_metrics.get("brightness", 0.5))
    vibrance = float(cv_metrics.get("vibrance", 0.5))

    contrast_norm = clamp01(contrast / 60.0)
    brightness_inv = 1.0 - clamp01(brightness)

    target_valence = clamp01(0.5 + 0.5 * nlp_valence + 0.1 * temperature)
    target_arousal = clamp01(0.5 + 0.45 * nlp_arousal + 0.15 * contrast_norm + 0.1 * (vibrance - 0.5))
    target_tension = clamp01(nlp_tension + 0.2 * contrast_norm + 0.2 * brightness_inv)
    target_energy = clamp01(0.45 + 0.35 * target_arousal + 0.2 * target_tension)
    target_tempo = 60.0 + 120.0 * target_arousal + 20.0 * target_tension
    target_brightness = clamp01(0.4 + 0.35 * (temperature + 1.0) / 2.0 + 0.25 * contrast_norm)
    target_loudness = -23.0 + 10.0 * target_energy + 4.0 * target_tension
    target_dynamic = clamp01(0.7 - 0.4 * target_tension + 0.1 * (1.0 - target_energy))
    target_percussive = clamp01(0.35 + 0.45 * target_arousal + 0.1 * target_tension)
    target_rhythm = clamp01(0.3 + 0.5 * target_tension + 0.2 * target_arousal)
    target_timbral = clamp01(0.35 + 0.45 * target_tension + 0.2 * target_arousal)

    return {
        "valence": target_valence,
        "arousal": target_arousal,
        "tension": target_tension,
        "energy": target_energy,
        "tempo": float(min(190.0, max(55.0, target_tempo))),
        "brightness": target_brightness,
        "loudness": target_loudness,
        "dynamic_range": target_dynamic,
        "percussive": target_percussive,
        "rhythm_complexity": target_rhythm,
        "timbral": target_timbral,
    }


def _genre_targets(scene: dict, cfg: dict) -> dict[str, float]:
    rules = cfg.get("genre_rules", {})
    emotion_rules = rules.get("emotion", {})
    mood_rules = rules.get("mood", {})
    setting_rules = rules.get("setting", {})

    nlp = scene["nlp"]
    emotion_profile = nlp.get("emotion_profile", {})
    mood_tags = nlp.get("mood_tags", [])
    setting_tags = nlp.get("setting_tags", [])

    scores: dict[str, float] = {}

    for emotion, weight in emotion_profile.items():
        if weight <= 0:
            continue
        for genre in emotion_rules.get(emotion, []):
            scores[genre] = scores.get(genre, 0.0) + float(weight)

    for tag in mood_tags:
        for genre in mood_rules.get(tag, []):
            scores[genre] = scores.get(genre, 0.0) + 0.2

    for tag in setting_tags:
        for genre in setting_rules.get(tag, []):
            scores[genre] = scores.get(genre, 0.0) + 0.15

    total = sum(scores.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in scores.items()}


def _genre_match_score(track_genres: list[str], genre_targets: dict[str, float]) -> tuple[float, list[str]]:
    if not genre_targets:
        return 0.5, []
    if not track_genres:
        return 0.0, []

    lowered = [g.lower() for g in track_genres]
    matched: list[str] = []
    score = 0.0

    for target, weight in genre_targets.items():
        t = target.lower()
        has_match = any((t in g) or (g in t) for g in lowered)
        if has_match:
            matched.append(target)
            score += weight

    return clamp01(score), matched


def _genre_dist_score(genre_dist: dict[str, float], genre_targets: dict[str, float]) -> float:
    if not genre_targets:
        return 0.5
    if not genre_dist:
        return 0.0
    score = 0.0
    for genre, weight in genre_targets.items():
        score += float(weight) * float(genre_dist.get(genre, 0.0))
    return clamp01(score)


def _audio_match_score(track: dict, target: dict, weights: dict) -> tuple[float, dict]:
    valence_sim = 1.0 - abs(float(track.get("valence_proxy", 0.5)) - target["valence"])
    arousal_sim = 1.0 - abs(float(track.get("arousal_proxy", 0.5)) - target["arousal"])
    tension_sim = 1.0 - abs(float(track.get("tension_proxy", 0.5)) - target["tension"])
    energy_sim = 1.0 - abs(float(track.get("energy_proxy", 0.5)) - target["energy"])
    tempo_sim = 1.0 - min(abs(float(track.get("tempo", 120.0)) - target["tempo"]) / 130.0, 1.0)
    brightness_sim = 1.0 - abs(float(track.get("brightness_proxy", 0.5)) - target["brightness"])
    loudness_sim = 1.0 - min(abs(float(track.get("loudness_lufs", -23.0)) - target["loudness"]) / 20.0, 1.0)
    dynamic_range_norm = clamp01(float(track.get("dynamic_range", 0.0)) / 0.15)
    dynamic_sim = 1.0 - abs(dynamic_range_norm - target["dynamic_range"])
    percussive = float(track.get("percussive_share", 0.0))
    hpr_sim = 1.0 - abs(percussive - target["percussive"])
    rhythm_sim = 1.0 - abs(float(track.get("rhythm_complexity", 0.0)) - target["rhythm_complexity"])
    timbral_complexity = float(track.get("timbral_stats", {}).get("timbral_complexity", 0.0))
    timbral_sim = 1.0 - abs(timbral_complexity - target["timbral"])

    sims = {
        "valence": clamp01(valence_sim),
        "arousal": clamp01(arousal_sim),
        "tension": clamp01(tension_sim),
        "energy": clamp01(energy_sim),
        "tempo": clamp01(tempo_sim),
        "brightness": clamp01(brightness_sim),
        "loudness": clamp01(loudness_sim),
        "dynamic_range": clamp01(dynamic_sim),
        "hpr": clamp01(hpr_sim),
        "rhythm_complexity": clamp01(rhythm_sim),
        "timbral": clamp01(timbral_sim),
    }
    total_w = sum(weights.values()) or 1.0
    score = sum(sims[k] * float(weights.get(k, 0.0)) for k in sims) / total_w
    return clamp01(score), sims


def build_scene_features(scene: dict, base_dir: Path, cfg: dict) -> dict:
    tokens = tokenize(scene["context"])
    nlp_features = compute_scene_features(tokens)

    cv_metrics = {}
    if scene.get("frames_path"):
        frames_path = Path(scene["frames_path"])
        if not frames_path.is_absolute():
            frames_path = (base_dir / frames_path).resolve()
        cv_cfg = cfg.get("cv", {})
        cv_result = extract_palette(
            input_path=frames_path,
            k=int(cv_cfg.get("k", 5)),
            resize=int(cv_cfg.get("resize", 200)),
            max_images=int(cv_cfg.get("max_images", 0)),
            stride=int(cv_cfg.get("stride", 1)),
            max_pixels=int(cv_cfg.get("max_pixels", 5000)),
            sample_total=int(cv_cfg.get("sample_total", 200000)),
            window_size=int(cv_cfg.get("window_size", 0)),
            window_step=int(cv_cfg.get("window_step", 0)),
            fps=float(cv_cfg.get("fps", 0.0)),
            seed=int(cv_cfg.get("seed", 7)),
        )
        cv_metrics = aggregate_cv_metrics(cv_result)

    target = _emotion_target(nlp_features, cv_metrics)
    built = {"id": scene["id"], "context": scene["context"], "nlp": nlp_features, "cv": cv_metrics, "target": target}
    built["genre_targets"] = _genre_targets(built, cfg)
    return built


def _scene_tag_vector(genre_targets: dict[str, float], tags: list[str], genre_map: dict[str, list[str]]) -> np.ndarray:
    if not genre_targets or not tags or not genre_map:
        return np.zeros(len(tags), dtype=float)
    vec = np.zeros(len(tags), dtype=float)
    for idx, tag in enumerate(tags):
        mapped = genre_map.get(tag, [])
        if not mapped:
            continue
        for genre in mapped:
            vec[idx] += float(genre_targets.get(genre, 0.0))
    total = vec.sum()
    if total > 0:
        vec = vec / total
    return vec


def _embedding_projection(tracks: list[dict]) -> tuple[np.ndarray | None, list[str]]:
    usable = [t for t in tracks if "audio_embedding" in t and "audio_tag_probs" in t and "audio_tags" in t]
    if not usable:
        return None, []

    tags = usable[0]["audio_tags"]
    tag_len = len(tags)
    embeddings = []
    tag_probs = []
    for t in usable:
        if t.get("audio_tags") != tags:
            continue
        emb = np.array(t["audio_embedding"], dtype=float)
        probs = np.array(t["audio_tag_probs"], dtype=float)
        if probs.size != tag_len:
            continue
        embeddings.append(emb)
        tag_probs.append(probs)

    if len(embeddings) < 2:
        return None, tags

    T = np.vstack(tag_probs)
    E = np.vstack(embeddings)
    try:
        W, _, _, _ = np.linalg.lstsq(T, E, rcond=None)
    except Exception:
        return None, tags
    return W, tags


def recommend_for_scene(scene: dict, tracks: list[dict], cfg: dict, top_k: int, genre_map: dict[str, list[str]], emb_proj: np.ndarray | None, emb_tags: list[str]) -> dict:
    matching = cfg.get("matching", {})
    weights = matching.get("feature_weights", {})
    final_weights = matching.get("weights", {})
    w_audio = float(final_weights.get("audio", 0.75))
    w_genre = float(final_weights.get("genre", 0.2))
    w_genre_audio = float(final_weights.get("genre_audio", 0.1))
    w_embed = float(final_weights.get("embedding", 0.1))

    scene_tag_vec = _scene_tag_vector(scene["genre_targets"], emb_tags, genre_map) if emb_tags else None
    scene_embedding = None
    if emb_proj is not None and scene_tag_vec is not None and scene_tag_vec.size:
        scene_embedding = scene_tag_vec @ emb_proj

    ranked = []
    for track in tracks:
        audio_score, sims = _audio_match_score(track, scene["target"], weights)
        spotify_score, matched = _genre_match_score(track.get("spotify_genres", []), scene["genre_targets"])
        audio_genre_score = _genre_dist_score(track.get("audio_genre_dist", {}), scene["genre_targets"])
        embedding_similarity = 0.0
        if scene_embedding is not None and "audio_embedding" in track:
            emb = np.array(track["audio_embedding"], dtype=float)
            denom = (np.linalg.norm(scene_embedding) * np.linalg.norm(emb))
            if denom > 0:
                embedding_similarity = float(np.dot(scene_embedding, emb) / denom)
                embedding_similarity = clamp01((embedding_similarity + 1.0) / 2.0)

        total_weight = w_audio + w_genre + w_genre_audio + w_embed
        if total_weight <= 0:
            total_weight = 1.0
        final_score = (
            w_audio * audio_score
            + w_genre * spotify_score
            + w_genre_audio * audio_genre_score
            + w_embed * embedding_similarity
        ) / total_weight
        final_score = clamp01(final_score)
        ranked.append(
            {
                "path": track.get("path", ""),
                "title": track.get("title", ""),
                "artist": track.get("artist", ""),
                "score": final_score,
                "audio_score": audio_score,
                "genre_score": spotify_score,
                "audio_genre_score": audio_genre_score,
                "matched_genres": matched,
                "spotify_genres": track.get("spotify_genres", []),
                "audio_features": {
                    "tempo": track.get("tempo"),
                    "key": track.get("key"),
                    "mode": track.get("mode"),
                    "energy_proxy": track.get("energy_proxy"),
                    "valence_proxy": track.get("valence_proxy"),
                    "arousal_proxy": track.get("arousal_proxy"),
                    "tension_proxy": track.get("tension_proxy"),
                    "danceability_proxy": track.get("danceability_proxy"),
                    "brightness_proxy": track.get("brightness_proxy"),
                    "loudness_lufs": track.get("loudness_lufs"),
                    "harmonic_percussive_ratio": track.get("harmonic_percussive_ratio"),
                    "rhythm_complexity": track.get("rhythm_complexity"),
                    "timbral_stats": track.get("timbral_stats"),
                    "segment_stats": track.get("segment_stats"),
                    "embedding_similarity": embedding_similarity,
                },
                "component_similarity": sims,
            }
        )

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return {
        "scene_id": scene["id"],
        "context": scene["context"],
        "target": scene["target"],
        "genre_targets": scene["genre_targets"],
        "recommendations": ranked[:top_k],
    }


def load_genre_map(path: Path | None) -> dict[str, list[str]]:
    if not path or not path.exists():
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        return {}
    return {str(k): list(v) for k, v in obj.items() if isinstance(v, list)}


def run_recommendation_pipeline(
    scenes_json: Path,
    audio_dir: Path,
    metadata_csv: Path | None,
    config_path: Path | None,
    include_spotify_genres: bool,
    include_audio_genre: bool,
    include_embeddings: bool,
    embedding_backend: str,
    top_k: int,
    cache_path: Path | None,
    embedding_cache_path: Path | None,
    genre_map_path: Path | None,
) -> dict:
    cfg = load_config(config_path)
    scenes_raw = load_scene_contexts(scenes_json)
    if not scenes_raw:
        raise ValueError("No scenes found in scenes JSON.")

    audio_cfg = cfg.get("audio", {})
    tracks = analyze_library(
        audio_dir=audio_dir,
        metadata_csv=metadata_csv,
        sample_seconds=float(audio_cfg.get("sample_seconds", 90.0)),
        sample_offset=float(audio_cfg.get("sample_offset", 0.0)),
        sr=int(audio_cfg.get("sr", 22050)),
        segment_count=int(audio_cfg.get("segment_count", 3)),
        segment_strategy=str(audio_cfg.get("segment_strategy", "equal")),
        include_spotify_genres=include_spotify_genres,
        include_embeddings=include_embeddings,
        embedding_backend=embedding_backend or str(audio_cfg.get("embedding_backend", "musicnn")),
        embedding_dim=int(audio_cfg.get("embedding_dim", 200)),
        cache_version=int(audio_cfg.get("cache_version", 2)),
        spotify_market=str(audio_cfg.get("spotify_market", "US")),
        cache_path=cache_path,
        embedding_cache_path=embedding_cache_path,
        genre_map_path=genre_map_path if include_audio_genre else None,
    )

    if not tracks:
        raise ValueError("No audio files found.")

    base_dir = scenes_json.parent.resolve()
    built_scenes = [build_scene_features(scene, base_dir, cfg) for scene in scenes_raw]
    genre_map = load_genre_map(genre_map_path)
    emb_proj, emb_tags = _embedding_projection(tracks) if include_embeddings else (None, [])
    recommendations = [
        recommend_for_scene(scene, tracks, cfg, top_k, genre_map, emb_proj, emb_tags)
        for scene in built_scenes
    ]

    return {
        "num_scenes": len(recommendations),
        "library_size": len(tracks),
        "scenes": recommendations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend songs for scene context using NLP + CV + audio + Spotify genres.")
    parser.add_argument("--scenes-json", required=True, help="Scene context JSON path")
    parser.add_argument("--audio-dir", required=True, help="Directory of candidate MP3/audio files")
    parser.add_argument("--metadata-csv", default="", help="Optional metadata CSV: path,title,artist")
    parser.add_argument("--config", default="configs/pipeline.json", help="Pipeline config JSON")
    parser.add_argument("--top-k", type=int, default=10, help="Top recommendations per scene")
    parser.add_argument("--no-spotify-genres", action="store_true", help="Disable Spotify genre lookup")
    parser.add_argument("--no-audio-genre", action="store_true", help="Disable audio genre scoring")
    parser.add_argument("--no-embeddings", action="store_true", help="Disable audio embeddings")
    parser.add_argument("--embedding-backend", default="", help="Override embedding backend")
    parser.add_argument("--cache", default="data/audio_cache.json", help="Audio analysis cache path")
    parser.add_argument("--embedding-cache", default="data/audio_embeddings.npz", help="Embedding cache path")
    parser.add_argument("--genre-map", default="configs/genre_map.json", help="Audio genre map JSON")
    parser.add_argument("--out", default="data/recommendations.json", help="Output JSON path")

    args = parser.parse_args()
    result = run_recommendation_pipeline(
        scenes_json=Path(args.scenes_json),
        audio_dir=Path(args.audio_dir),
        metadata_csv=Path(args.metadata_csv) if args.metadata_csv else None,
        config_path=Path(args.config) if args.config else None,
        include_spotify_genres=not args.no_spotify_genres,
        include_audio_genre=not args.no_audio_genre,
        include_embeddings=not args.no_embeddings,
        embedding_backend=args.embedding_backend,
        top_k=args.top_k,
        cache_path=Path(args.cache) if args.cache else None,
        embedding_cache_path=Path(args.embedding_cache) if args.embedding_cache else None,
        genre_map_path=Path(args.genre_map) if args.genre_map else None,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote recommendations to: {out_path}")


if __name__ == "__main__":
    main()
