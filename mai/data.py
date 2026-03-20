import pandas as pd


def load_csv_playlist(path: str) -> pd.DataFrame:
    """Load a playlist CSV (or similar) into a DataFrame.

    This function does minimal cleaning (strips column names).
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def ensure_audio_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure common audio feature columns exist (may coerce types)."""
    numeric_cols = [
        'danceability','energy','speechiness','acousticness','liveness','valence',
        'tempo','key','mode','loudness','rms','spectral_centroid','spectral_bandwidth',
        'spectral_flatness','spectral_rolloff','zcr','onset_strength','harmonic_ratio'
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Also coerce derived intro_/outro_/sentiment variants if present in CSV.
    extra_bases = numeric_cols + ['intro_seconds_used', 'outro_seconds_used', 'intro_leading_silence_s', 'outro_trailing_silence_s']
    for col in list(df.columns):
        lower = str(col).strip().lower()
        if lower in extra_bases or lower in {'genre_confidence'} or lower.startswith('intro_') or lower.startswith('outro_') or lower.startswith('sentiment_'):
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def normalize_audio_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common audio feature column names to lowercase for consistency."""
    aliases = {
        'danceability': 'danceability',
        'energy': 'energy',
        'speechiness': 'speechiness',
        'acousticness': 'acousticness',
        'liveness': 'liveness',
        'valence': 'valence',
        'tempo': 'tempo',
        'key': 'key',
        'mode': 'mode',
        'loudness': 'loudness',
        'rms': 'rms',
        'spectral_centroid': 'spectral_centroid',
        'spectral_bandwidth': 'spectral_bandwidth',
        'spectral_flatness': 'spectral_flatness',
        'spectral_rolloff': 'spectral_rolloff',
        'zcr': 'zcr',
        'onset_strength': 'onset_strength',
        'harmonic_ratio': 'harmonic_ratio',
    }
    rename_map = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in aliases and col != aliases[key]:
            rename_map[col] = aliases[key]
    if rename_map:
        df = df.rename(columns=rename_map)
    return df
