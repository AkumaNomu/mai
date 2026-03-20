import re
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

from .sentiment import SENTIMENT_DIMS, add_sentiment_features


CANONICAL_GENRES = [
    'pop',
    'rock',
    'indie_alt',
    'electronic',
    'house_techno',
    'hip_hop_rap',
    'rnb_soul',
    'jazz',
    'ambient_chill',
    'folk_acoustic',
    'latin_world',
    'classical_score',
    'metal_punk',
    'unknown',
]

METADATA_FIELDS = ['title', 'artist', 'uploader', 'channel', 'tags', 'category', 'description']
GENRE_SOURCE_MIN_CONFIDENCE = 0.55

GENRE_KEYWORDS = {
    'pop': ['pop', 'dance pop', 'synthpop', 'teen pop', 'electropop'],
    'rock': ['rock', 'hard rock', 'arena rock', 'classic rock', 'grunge'],
    'indie_alt': ['indie', 'alternative', 'alt', 'shoegaze', 'dream pop', 'post punk', 'new wave', 'lofi indie'],
    'electronic': ['electronic', 'edm', 'electro', 'synthwave', 'drum and bass', 'dnb', 'breakbeat', 'electronica', 'downtempo'],
    'house_techno': ['house', 'techno', 'deep house', 'tech house', 'minimal', 'progressive house', 'trance'],
    'hip_hop_rap': ['hip hop', 'hip-hop', 'rap', 'trap', 'boom bap', 'drill'],
    'rnb_soul': ['rnb', 'r&b', 'soul', 'neo soul', 'motown', 'funk'],
    'jazz': ['jazz', 'bebop', 'swing', 'fusion', 'bossa', 'sax', 'blue note'],
    'ambient_chill': ['ambient', 'chill', 'chillout', 'new age', 'drone', 'meditation', 'lofi', 'study beats'],
    'folk_acoustic': ['folk', 'acoustic', 'singer songwriter', 'singer-songwriter', 'country folk', 'americana'],
    'latin_world': ['latin', 'reggaeton', 'afrobeats', 'afrobeat', 'salsa', 'cumbia', 'brazil', 'bossa nova', 'flamenco', 'world'],
    'classical_score': ['classical', 'orchestral', 'score', 'soundtrack', 'film score', 'piano', 'instrumental score', 'opera'],
    'metal_punk': ['metal', 'punk', 'hardcore', 'emo', 'screamo', 'thrash', 'death metal'],
}

GENRE_PROTOTYPES = {
    'pop': {'danceability': 0.65, 'energy': 0.60, 'acousticness': 0.20, 'speechiness': 0.12, 'tempo_norm': 0.55, 'sentiment_valence': 0.65, 'sentiment_arousal': 0.62, 'sentiment_warmth': 0.45, 'harmonic_ratio': 0.45},
    'rock': {'danceability': 0.45, 'energy': 0.78, 'acousticness': 0.15, 'speechiness': 0.08, 'tempo_norm': 0.55, 'sentiment_valence': 0.48, 'sentiment_arousal': 0.72, 'sentiment_warmth': 0.35, 'harmonic_ratio': 0.38},
    'indie_alt': {'danceability': 0.42, 'energy': 0.58, 'acousticness': 0.28, 'speechiness': 0.08, 'tempo_norm': 0.46, 'sentiment_valence': 0.42, 'sentiment_arousal': 0.50, 'sentiment_warmth': 0.48, 'harmonic_ratio': 0.50},
    'electronic': {'danceability': 0.72, 'energy': 0.74, 'acousticness': 0.08, 'speechiness': 0.08, 'tempo_norm': 0.63, 'sentiment_valence': 0.56, 'sentiment_arousal': 0.76, 'sentiment_warmth': 0.28, 'harmonic_ratio': 0.32},
    'house_techno': {'danceability': 0.82, 'energy': 0.80, 'acousticness': 0.04, 'speechiness': 0.05, 'tempo_norm': 0.72, 'sentiment_valence': 0.56, 'sentiment_arousal': 0.82, 'sentiment_warmth': 0.22, 'harmonic_ratio': 0.25},
    'hip_hop_rap': {'danceability': 0.76, 'energy': 0.62, 'acousticness': 0.12, 'speechiness': 0.45, 'tempo_norm': 0.48, 'sentiment_valence': 0.46, 'sentiment_arousal': 0.66, 'sentiment_warmth': 0.34, 'harmonic_ratio': 0.30},
    'rnb_soul': {'danceability': 0.62, 'energy': 0.48, 'acousticness': 0.18, 'speechiness': 0.16, 'tempo_norm': 0.42, 'sentiment_valence': 0.58, 'sentiment_arousal': 0.48, 'sentiment_warmth': 0.72, 'harmonic_ratio': 0.60},
    'jazz': {'danceability': 0.38, 'energy': 0.36, 'acousticness': 0.66, 'speechiness': 0.08, 'tempo_norm': 0.44, 'sentiment_valence': 0.54, 'sentiment_arousal': 0.34, 'sentiment_warmth': 0.68, 'harmonic_ratio': 0.78},
    'ambient_chill': {'danceability': 0.28, 'energy': 0.22, 'acousticness': 0.52, 'speechiness': 0.05, 'tempo_norm': 0.24, 'sentiment_valence': 0.50, 'sentiment_arousal': 0.18, 'sentiment_warmth': 0.72, 'harmonic_ratio': 0.70},
    'folk_acoustic': {'danceability': 0.36, 'energy': 0.30, 'acousticness': 0.82, 'speechiness': 0.08, 'tempo_norm': 0.34, 'sentiment_valence': 0.58, 'sentiment_arousal': 0.26, 'sentiment_warmth': 0.78, 'harmonic_ratio': 0.76},
    'latin_world': {'danceability': 0.74, 'energy': 0.68, 'acousticness': 0.28, 'speechiness': 0.14, 'tempo_norm': 0.62, 'sentiment_valence': 0.68, 'sentiment_arousal': 0.72, 'sentiment_warmth': 0.52, 'harmonic_ratio': 0.55},
    'classical_score': {'danceability': 0.12, 'energy': 0.24, 'acousticness': 0.90, 'speechiness': 0.02, 'tempo_norm': 0.26, 'sentiment_valence': 0.42, 'sentiment_arousal': 0.22, 'sentiment_warmth': 0.62, 'harmonic_ratio': 0.88},
    'metal_punk': {'danceability': 0.34, 'energy': 0.90, 'acousticness': 0.04, 'speechiness': 0.12, 'tempo_norm': 0.68, 'sentiment_valence': 0.28, 'sentiment_arousal': 0.88, 'sentiment_warmth': 0.18, 'harmonic_ratio': 0.22},
}

STYLE_CLUSTER_FEATURES = [
    'danceability', 'energy', 'speechiness', 'acousticness', 'liveness', 'valence',
    'log_tempo', 'spectral_centroid', 'spectral_bandwidth', 'spectral_flatness',
    'harmonic_ratio', 'sentiment_valence', 'sentiment_arousal', 'sentiment_tension', 'sentiment_warmth',
]


def _normalize_text(value) -> str:
    if pd.isna(value):
        return ''
    if isinstance(value, (list, tuple, set)):
        value = ' '.join(str(item) for item in value if item is not None)
    text = str(value).strip().lower()
    text = re.sub(r'[_\-]+', ' ', text)
    text = re.sub(r'[^a-z0-9& ]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _match_text_to_genre(text: str) -> tuple[str, float]:
    if not text:
        return 'unknown', 0.0
    scores = defaultdict(float)
    padded = f' {text} '
    for genre, keywords in GENRE_KEYWORDS.items():
        for keyword in keywords:
            token = _normalize_text(keyword)
            if not token:
                continue
            token_padded = f' {token} '
            if token_padded in padded:
                scores[genre] += 1.0 + 0.15 * max(len(token.split()) - 1, 0)
    if not scores:
        return 'unknown', 0.0
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_genre, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    confidence = float(np.clip(0.60 + 0.12 * best_score + 0.08 * (best_score - second_score), 0.0, 0.98))
    return best_genre, confidence


def _series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors='coerce').fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _norm(values: pd.Series, vmin: float, vmax: float) -> pd.Series:
    if vmax <= vmin:
        return pd.Series(0.0, index=values.index, dtype=float)
    return ((values - vmin) / (vmax - vmin)).clip(0.0, 1.0)


def _style_clusters(df: pd.DataFrame, genre_clusters: int = 8) -> pd.Series:
    feature_columns = [column for column in STYLE_CLUSTER_FEATURES if column in df.columns]
    if not feature_columns or len(df) == 0:
        return pd.Series(['cluster_00'] * len(df), index=df.index, dtype=str)
    if len(df) == 1:
        return pd.Series(['cluster_00'], index=df.index, dtype=str)
    matrix = df[feature_columns].fillna(0).to_numpy(dtype=float)
    matrix = RobustScaler().fit_transform(matrix)
    cluster_count = int(min(max(2, genre_clusters), len(df)))
    labels = KMeans(n_clusters=cluster_count, n_init=10, random_state=42).fit_predict(matrix)
    return pd.Series([f'cluster_{int(label):02d}' for label in labels], index=df.index, dtype=str)


def _metadata_genre_match(row: pd.Series) -> tuple[str, float]:
    field_weights = {
        'genre': 1.00,
        'genres': 1.00,
        'tags': 0.95,
        'category': 0.90,
        'title': 0.75,
        'artist': 0.55,
        'uploader': 0.50,
        'channel': 0.45,
        'description': 0.35,
    }
    scores = defaultdict(float)
    for field in METADATA_FIELDS + ['genre', 'genres']:
        if field not in row.index:
            continue
        genre, confidence = _match_text_to_genre(_normalize_text(row.get(field)))
        if genre == 'unknown':
            continue
        scores[genre] += confidence * field_weights.get(field, 0.5)
    if not scores:
        return 'unknown', 0.0
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_genre, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    confidence = float(np.clip(0.50 + 0.20 * best_score + 0.15 * (best_score - second_score), 0.0, 0.95))
    return best_genre, confidence


def _audio_genre_candidates(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    danceability = _series(df, 'danceability')
    energy = _series(df, 'energy')
    acousticness = _series(df, 'acousticness')
    speechiness = _series(df, 'speechiness')
    harmonic_ratio = _series(df, 'harmonic_ratio')
    sentiment_valence = _series(df, 'sentiment_valence')
    sentiment_arousal = _series(df, 'sentiment_arousal')
    sentiment_warmth = _series(df, 'sentiment_warmth')
    tempo_norm = _norm(_series(df, 'tempo'), 60.0, 180.0)

    feature_frame = pd.DataFrame({
        'danceability': danceability,
        'energy': energy,
        'acousticness': acousticness,
        'speechiness': speechiness,
        'harmonic_ratio': harmonic_ratio,
        'sentiment_valence': sentiment_valence,
        'sentiment_arousal': sentiment_arousal,
        'sentiment_warmth': sentiment_warmth,
        'tempo_norm': tempo_norm,
    }, index=df.index)

    best_genres = []
    confidences = []
    for _, row in feature_frame.iterrows():
        scores = {}
        for genre, prototype in GENRE_PROTOTYPES.items():
            diff = 0.0
            count = 0
            for key, target in prototype.items():
                if key not in row.index:
                    continue
                diff += abs(float(row[key]) - float(target))
                count += 1
            score = float(np.exp(-diff / max(count, 1)))
            scores[genre] = score
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_genre, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        confidence = float(np.clip(0.30 + 0.45 * best_score + 0.35 * (best_score - second_score), 0.0, 0.90))
        best_genres.append(best_genre)
        confidences.append(confidence)
    return pd.Series(best_genres, index=df.index, dtype=str), pd.Series(confidences, index=df.index, dtype=float)


def resolve_genres(df: pd.DataFrame, genre_column: str | None = None, genre_clusters: int = 8) -> pd.DataFrame:
    """Resolve mostly-local genre families and style clusters for generation and reporting."""
    df = add_sentiment_features(df.copy())
    df['style_cluster'] = _style_clusters(df, genre_clusters=genre_clusters)

    genre_primary = pd.Series(['unknown'] * len(df), index=df.index, dtype=str)
    genre_confidence = pd.Series([0.0] * len(df), index=df.index, dtype=float)
    genre_source = pd.Series(['unresolved'] * len(df), index=df.index, dtype=str)

    explicit_columns = []
    if genre_column and genre_column in df.columns:
        explicit_columns.append(genre_column)
    for candidate in ['genre', 'genres']:
        if candidate in df.columns and candidate not in explicit_columns:
            explicit_columns.append(candidate)

    for column in explicit_columns:
        for index, value in df[column].items():
            matched_genre, confidence = _match_text_to_genre(_normalize_text(value))
            if matched_genre != 'unknown' and genre_source.loc[index] == 'unresolved':
                genre_primary.loc[index] = matched_genre
                genre_confidence.loc[index] = min(1.0, max(confidence, 0.92))
                genre_source.loc[index] = 'input_column'

    for index, row in df.iterrows():
        if genre_source.loc[index] == 'input_column':
            continue
        metadata_genre, metadata_confidence = _metadata_genre_match(row)
        if metadata_genre != 'unknown' and genre_source.loc[index] == 'unresolved':
            genre_primary.loc[index] = metadata_genre
            genre_confidence.loc[index] = metadata_confidence
            genre_source.loc[index] = 'metadata_heuristic'

    audio_genre, audio_confidence = _audio_genre_candidates(df)
    for index in df.index:
        if audio_genre.loc[index] != 'unknown' and genre_source.loc[index] == 'unresolved':
            genre_primary.loc[index] = audio_genre.loc[index]
            genre_confidence.loc[index] = audio_confidence.loc[index]
            genre_source.loc[index] = 'audio_heuristic'

    low_confidence = genre_confidence < GENRE_SOURCE_MIN_CONFIDENCE
    genre_primary.loc[low_confidence] = 'unknown'
    df['genre_primary'] = genre_primary
    df['genre_confidence'] = genre_confidence
    df['genre_source'] = genre_source
    df['mix_group'] = np.where(genre_confidence >= GENRE_SOURCE_MIN_CONFIDENCE, genre_primary, df['style_cluster'])
    return df
