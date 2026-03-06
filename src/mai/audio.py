import argparse
import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from .audio_embeddings import EmbeddingCache, build_audio_genre_dist, extract_embedding
from .spotify import SpotifyGenreResolver

try:
    import pyloudnorm as pyln
except Exception:  # pragma: no cover - optional dependency
    pyln = None


AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def norm_linear(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return clamp01((value - low) / (high - low))


def list_audio_files(audio_dir: Path) -> list[Path]:
    if audio_dir.is_file() and audio_dir.suffix.lower() in AUDIO_EXTS:
        return [audio_dir]
    if not audio_dir.is_dir():
        raise ValueError(f"Audio path not found: {audio_dir}")

    files: list[Path] = []
    for ext in AUDIO_EXTS:
        files.extend(audio_dir.rglob(f"*{ext}"))
    return sorted(files)


def infer_title_artist(path: Path) -> tuple[str, str]:
    stem = path.stem.strip()
    if " - " in stem:
        artist, title = stem.split(" - ", 1)
        return title.strip(), artist.strip()
    return stem, ""


def load_metadata_csv(metadata_csv: Path | None) -> dict[str, dict]:
    if not metadata_csv:
        return {}
    if not metadata_csv.exists():
        raise ValueError(f"Metadata CSV not found: {metadata_csv}")

    df = pd.read_csv(metadata_csv)
    required = {"path", "title", "artist"}
    if not required.issubset(set(df.columns)):
        raise ValueError("Metadata CSV must include columns: path,title,artist")

    lookup: dict[str, dict] = {}
    for _, row in df.iterrows():
        p = str(row["path"]).replace("\\", "/")
        lookup[p] = {"title": str(row["title"]), "artist": str(row["artist"])}
    return lookup


def detect_key_mode(chroma_mean: np.ndarray) -> tuple[int, str, float]:
    best_key = 0
    best_mode = "major"
    best_score = -1e9

    for key in range(12):
        major_score = float(np.dot(chroma_mean, np.roll(MAJOR_PROFILE, key)))
        minor_score = float(np.dot(chroma_mean, np.roll(MINOR_PROFILE, key)))
        if major_score > best_score:
            best_score = major_score
            best_key = key
            best_mode = "major"
        if minor_score > best_score:
            best_score = minor_score
            best_key = key
            best_mode = "minor"
    return best_key, best_mode, best_score


def split_segments(y: np.ndarray, segment_count: int) -> list[np.ndarray]:
    count = max(1, int(segment_count))
    if count == 1 or y.size == 0:
        return [y]

    seg_len = int(len(y) / count)
    segments: list[np.ndarray] = []
    for idx in range(count):
        start = idx * seg_len
        end = len(y) if idx == count - 1 else (idx + 1) * seg_len
        segments.append(y[start:end])
    return segments


def compute_lufs(y: np.ndarray, sr: int) -> float | None:
    if pyln is None or y.size == 0:
        return None
    try:
        meter = pyln.Meter(sr)
        return float(meter.integrated_loudness(y))
    except Exception:
        return None


def spectral_flux_mean(y: np.ndarray, sr: int) -> float:
    try:
        S = np.abs(librosa.stft(y))
        if S.shape[1] < 2:
            return 0.0
        diff = np.diff(S, axis=1)
        flux = np.sqrt(np.mean(diff**2, axis=0))
        return float(np.mean(flux))
    except Exception:
        return 0.0


def analyze_segment(y: np.ndarray, sr: int) -> dict[str, Any]:
    if y.size == 0:
        return {}

    duration = max(len(y) / sr, 1e-6)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_rate = float(len(onset_frames) / duration)

    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    harmonic, percussive = librosa.effects.hpss(y)
    h_energy = float(np.mean(harmonic**2))
    p_energy = float(np.mean(percussive**2))
    hpr = float(h_energy / (p_energy + 1e-9))
    percussive_share = float(p_energy / (h_energy + p_energy + 1e-9))

    rms_mean = float(rms.mean())
    dynamic_range = float(np.percentile(rms, 95) - np.percentile(rms, 5))
    centroid_mean = float(centroid.mean())
    rolloff_mean = float(rolloff.mean())
    flatness_mean = float(flatness.mean())
    contrast_mean = float(contrast.mean())
    onset_std = float(onset_env.std()) if onset_env.size else 0.0
    zcr_mean = float(zcr.mean())
    flux_mean = spectral_flux_mean(y, sr)

    beat_regularity = 0.0
    if len(beat_frames) > 3:
        beat_intervals = np.diff(beat_frames)
        beat_regularity = clamp01(1.0 - norm_linear(float(np.std(beat_intervals)), 0.0, 12.0))

    onset_norm = norm_linear(onset_std, 0.5, 8.0)
    rhythm_complexity = clamp01(onset_norm * (1.1 - beat_regularity))

    lufs = compute_lufs(y, sr)

    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)
    timbral_complexity = clamp01(float(np.mean(mfcc_std[:5])) / 20.0)

    return {
        "tempo": float(tempo),
        "onset_rate": onset_rate,
        "beat_regularity": float(beat_regularity),
        "rms_mean": rms_mean,
        "dynamic_range": dynamic_range,
        "spectral_centroid_mean": centroid_mean,
        "spectral_rolloff_mean": rolloff_mean,
        "spectral_flatness_mean": flatness_mean,
        "spectral_contrast_mean": contrast_mean,
        "spectral_flux_mean": flux_mean,
        "harmonic_percussive_ratio": hpr,
        "percussive_share": percussive_share,
        "chroma_mean": [float(v) for v in chroma.mean(axis=1)],
        "tonnetz_mean": [float(v) for v in tonnetz.mean(axis=1)],
        "mfcc_mean": [float(v) for v in mfcc_mean],
        "mfcc_std": [float(v) for v in mfcc_std],
        "timbral_complexity": timbral_complexity,
        "loudness_lufs": lufs,
        "zcr_mean": zcr_mean,
        "onset_std": onset_std,
        "duration_sec": duration,
    }


def aggregate_segments(segment_stats: list[dict[str, Any]]) -> dict[str, Any]:
    if not segment_stats:
        return {}

    numeric_keys = {
        "tempo",
        "onset_rate",
        "beat_regularity",
        "rms_mean",
        "dynamic_range",
        "spectral_centroid_mean",
        "spectral_rolloff_mean",
        "spectral_flatness_mean",
        "spectral_contrast_mean",
        "spectral_flux_mean",
        "harmonic_percussive_ratio",
        "percussive_share",
        "timbral_complexity",
        "loudness_lufs",
        "zcr_mean",
        "onset_std",
    }

    aggregated: dict[str, Any] = {}
    for key in numeric_keys:
        values = [s.get(key) for s in segment_stats if isinstance(s.get(key), (int, float))]
        if not values:
            continue
        aggregated[f"{key}_mean"] = float(np.mean(values))
        aggregated[f"{key}_median"] = float(np.median(values))
        aggregated[f"{key}_slope"] = float(values[-1] - values[0]) if len(values) > 1 else 0.0

    def avg_vector(field: str) -> list[float]:
        vectors = [s.get(field) for s in segment_stats if isinstance(s.get(field), list)]
        if not vectors:
            return []
        return [float(v) for v in np.mean(np.array(vectors), axis=0).tolist()]

    aggregated["mfcc_mean"] = avg_vector("mfcc_mean")
    aggregated["mfcc_std"] = avg_vector("mfcc_std")
    aggregated["chroma_mean"] = avg_vector("chroma_mean")
    aggregated["tonnetz_mean"] = avg_vector("tonnetz_mean")

    return aggregated


def compute_proxies(agg: dict[str, Any]) -> dict[str, float]:
    tempo = float(agg.get("tempo_mean", 120.0))
    rms_mean = float(agg.get("rms_mean_mean", 0.05))
    onset_std = float(agg.get("onset_std_mean", 1.0))
    centroid_mean = float(agg.get("spectral_centroid_mean_mean", 1200.0))
    flatness_mean = float(agg.get("spectral_flatness_mean_mean", 0.05))
    contrast_mean = float(agg.get("spectral_contrast_mean_mean", 20.0))

    tempo_norm = norm_linear(tempo, 60.0, 180.0)
    rms_norm = norm_linear(rms_mean, 0.01, 0.18)
    onset_norm = norm_linear(onset_std, 0.5, 8.0)
    centroid_norm = norm_linear(centroid_mean, 600.0, 4200.0)
    flatness_norm = norm_linear(flatness_mean, 0.005, 0.25)
    contrast_norm = norm_linear(contrast_mean, 10.0, 45.0)

    energy_proxy = clamp01(0.5 * rms_norm + 0.3 * onset_norm + 0.2 * tempo_norm)
    arousal_proxy = clamp01(0.35 * tempo_norm + 0.35 * onset_norm + 0.3 * rms_norm)
    tension_proxy = clamp01(0.4 * onset_norm + 0.35 * flatness_norm + 0.25 * contrast_norm)
    brightness_proxy = clamp01(centroid_norm)

    return {
        "energy_proxy": energy_proxy,
        "arousal_proxy": arousal_proxy,
        "tension_proxy": tension_proxy,
        "brightness_proxy": brightness_proxy,
    }


def analyze_audio_file(
    path: Path,
    sample_seconds: float,
    sample_offset: float,
    sr: int,
    segment_count: int,
    segment_strategy: str,
) -> dict[str, Any]:
    if segment_strategy != "equal":
        raise ValueError("Only segment_strategy='equal' is supported in v1.")

    y, loaded_sr = librosa.load(
        str(path),
        sr=sr,
        mono=True,
        duration=sample_seconds if sample_seconds > 0 else None,
        offset=sample_offset,
    )
    if y.size == 0:
        raise ValueError(f"Could not decode audio: {path}")

    chroma = librosa.feature.chroma_cqt(y=y, sr=loaded_sr)
    chroma_mean = chroma.mean(axis=1)
    key_idx, mode, key_conf = detect_key_mode(chroma_mean)

    segments = split_segments(y, segment_count)
    segment_stats = [analyze_segment(seg, loaded_sr) for seg in segments]
    agg = aggregate_segments(segment_stats)
    proxies = compute_proxies(agg)

    mode_boost = 0.2 if mode == "major" else -0.2
    valence_proxy = clamp01(0.5 + mode_boost + 0.18 * proxies["brightness_proxy"] - 0.18 * proxies["tension_proxy"])

    lufs = agg.get("loudness_lufs_mean")
    if lufs is None:
        rms_mean = agg.get("rms_mean_mean", 0.05)
        lufs = -20.0 * np.log10(max(rms_mean, 1e-6))

    return {
        "sample_rate": int(loaded_sr),
        "duration_analyzed_sec": float(len(y) / loaded_sr),
        "tempo": float(agg.get("tempo_mean", 0.0)),
        "key": NOTE_NAMES[key_idx],
        "key_index": int(key_idx),
        "mode": mode,
        "key_confidence": float(key_conf),
        "energy_proxy": proxies["energy_proxy"],
        "valence_proxy": valence_proxy,
        "arousal_proxy": proxies["arousal_proxy"],
        "tension_proxy": proxies["tension_proxy"],
        "danceability_proxy": clamp01(
            0.4 * float(agg.get("beat_regularity_mean", 0.0))
            + 0.3 * clamp01(1.0 - abs(float(agg.get("tempo_mean", 120.0)) - 120.0) / 120.0)
            + 0.3 * norm_linear(float(agg.get("rms_mean_mean", 0.05)), 0.01, 0.18)
        ),
        "brightness_proxy": proxies["brightness_proxy"],
        "rms_mean": float(agg.get("rms_mean_mean", 0.0)),
        "dynamic_range": float(agg.get("dynamic_range_mean", 0.0)),
        "zcr_mean": float(agg.get("zcr_mean_mean", 0.0)),
        "spectral_centroid_mean": float(agg.get("spectral_centroid_mean_mean", 0.0)),
        "spectral_rolloff_mean": float(agg.get("spectral_rolloff_mean_mean", 0.0)),
        "spectral_flatness_mean": float(agg.get("spectral_flatness_mean_mean", 0.0)),
        "spectral_contrast_mean": float(agg.get("spectral_contrast_mean_mean", 0.0)),
        "spectral_flux_mean": float(agg.get("spectral_flux_mean_mean", 0.0)),
        "onset_rate": float(agg.get("onset_rate_mean", 0.0)),
        "rhythm_complexity": float(agg.get("rhythm_complexity_mean", 0.0)),
        "harmonic_percussive_ratio": float(agg.get("harmonic_percussive_ratio_mean", 0.0)),
        "percussive_share": float(agg.get("percussive_share_mean", 0.0)),
        "timbral_stats": {
            "mfcc_mean": agg.get("mfcc_mean", []),
            "mfcc_std": agg.get("mfcc_std", []),
            "timbral_complexity": float(agg.get("timbral_complexity_mean", 0.0)),
        },
        "loudness_lufs": float(lufs),
        "segment_stats": segment_stats,
    }


def _cache_key(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def _cache_signature(
    cache_version: int,
    sample_seconds: float,
    sample_offset: float,
    sr: int,
    segment_count: int,
    segment_strategy: str,
) -> str:
    sig = {
        "cache_version": int(cache_version),
        "sample_seconds": float(sample_seconds),
        "sample_offset": float(sample_offset),
        "sr": int(sr),
        "segment_count": int(segment_count),
        "segment_strategy": segment_strategy,
    }
    return json.dumps(sig, sort_keys=True)


def analyze_library(
    audio_dir: Path,
    metadata_csv: Path | None = None,
    sample_seconds: float = 90.0,
    sample_offset: float = 0.0,
    sr: int = 22050,
    segment_count: int = 3,
    segment_strategy: str = "equal",
    include_spotify_genres: bool = True,
    include_embeddings: bool = True,
    embedding_backend: str = "musicnn",
    embedding_dim: int = 200,
    cache_version: int = 2,
    spotify_market: str = "US",
    cache_path: Path | None = None,
    embedding_cache_path: Path | None = None,
    genre_map_path: Path | None = None,
) -> list[dict]:
    files = list_audio_files(audio_dir)
    metadata_lookup = load_metadata_csv(metadata_csv)

    cached: dict[str, dict] = {}
    if cache_path and cache_path.exists():
        try:
            cache_obj = json.loads(cache_path.read_text(encoding="utf-8"))
            cached = cache_obj if isinstance(cache_obj, dict) else {}
        except Exception:
            cached = {}

    resolver = SpotifyGenreResolver(market=spotify_market) if include_spotify_genres else None

    emb_cache = None
    if include_embeddings and embedding_cache_path:
        emb_cache = EmbeddingCache(
            path=embedding_cache_path,
            backend=embedding_backend,
            version=cache_version,
            embedding_dim=embedding_dim,
        )

    signature = _cache_signature(cache_version, sample_seconds, sample_offset, sr, segment_count, segment_strategy)
    tracks: list[dict] = []

    for path in tqdm(files, desc="Analyzing audio", unit="track"):
        key = _cache_key(path)
        mtime = path.stat().st_mtime

        use_cached = False
        if key in cached:
            entry = cached[key]
            if (
                float(entry.get("_mtime", -1)) == float(mtime)
                and entry.get("_signature") == signature
            ):
                use_cached = True

        if use_cached:
            track = dict(entry["track"])
        else:
            rel = str(path).replace("\\", "/")
            meta = metadata_lookup.get(rel) or metadata_lookup.get(path.name)
            if meta:
                title = meta.get("title", "").strip() or path.stem
                artist = meta.get("artist", "").strip()
            else:
                title, artist = infer_title_artist(path)

            audio_features = analyze_audio_file(
                path=path,
                sample_seconds=sample_seconds,
                sample_offset=sample_offset,
                sr=sr,
                segment_count=segment_count,
                segment_strategy=segment_strategy,
            )
            track = {
                "path": str(path),
                "title": title,
                "artist": artist,
                "spotify_genres": [],
                **audio_features,
            }

            cached[key] = {
                "_mtime": mtime,
                "_signature": signature,
                "track": track,
            }

        if resolver and resolver.enabled:
            title = track.get("title", "")
            artist = track.get("artist", "")
            track["spotify_genres"] = resolver.lookup_genres(title=title, artist=artist)

        if emb_cache:
            emb_entry = emb_cache.get_or_compute(
                path=path,
                mtime=mtime,
                backend=embedding_backend,
            )
            if emb_entry:
                track["audio_embedding"] = emb_entry["embedding"]
                track["audio_tags"] = emb_entry["tags"]
                track["audio_tag_probs"] = emb_entry["tag_probs"]
                track["audio_genre_dist"] = build_audio_genre_dist(
                    tag_probs=emb_entry["tag_probs"],
                    tags=emb_entry["tags"],
                    genre_map_path=genre_map_path,
                )

        tracks.append(track)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cached, indent=2), encoding="utf-8")

    if emb_cache:
        emb_cache.save()

    return tracks


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze local audio files and infer musical features.")
    parser.add_argument("--audio-dir", required=True, help="Directory containing MP3/audio files")
    parser.add_argument("--metadata-csv", default="", help="Optional CSV with columns path,title,artist")
    parser.add_argument("--sample-seconds", type=float, default=90.0, help="Seconds per track to analyze")
    parser.add_argument("--sample-offset", type=float, default=0.0, help="Start offset in seconds")
    parser.add_argument("--sr", type=int, default=22050, help="Resample rate for analysis")
    parser.add_argument("--segment-count", type=int, default=3, help="Number of segments per track")
    parser.add_argument("--segment-strategy", default="equal", help="Segment strategy (equal only in v1)")
    parser.add_argument("--embedding-backend", default="musicnn", help="Embedding backend")
    parser.add_argument("--no-embeddings", action="store_true", help="Disable audio embeddings")
    parser.add_argument("--embedding-cache", default="data/audio_embeddings.npz", help="Embedding cache path")
    parser.add_argument("--cache-version", type=int, default=2, help="Audio feature cache version")
    parser.add_argument("--no-spotify-genres", action="store_true", help="Disable Spotify genre lookup")
    parser.add_argument("--spotify-market", default="US", help="Spotify market code")
    parser.add_argument("--cache", default="data/audio_cache.json", help="Audio feature cache path")
    parser.add_argument("--genre-map", default="configs/genre_map.json", help="Genre map JSON for audio tags")
    parser.add_argument("--out", default="", help="Optional JSON output path")

    args = parser.parse_args()
    tracks = analyze_library(
        audio_dir=Path(args.audio_dir),
        metadata_csv=Path(args.metadata_csv) if args.metadata_csv else None,
        sample_seconds=args.sample_seconds,
        sample_offset=args.sample_offset,
        sr=args.sr,
        segment_count=args.segment_count,
        segment_strategy=args.segment_strategy,
        include_spotify_genres=not args.no_spotify_genres,
        include_embeddings=not args.no_embeddings,
        embedding_backend=args.embedding_backend,
        embedding_dim=200,
        cache_version=args.cache_version,
        spotify_market=args.spotify_market,
        cache_path=Path(args.cache) if args.cache else None,
        embedding_cache_path=Path(args.embedding_cache) if args.embedding_cache else None,
        genre_map_path=Path(args.genre_map) if args.genre_map else None,
    )
    result = {"num_tracks": len(tracks), "tracks": tracks}
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
