import numpy as np
import pandas as pd


SENTIMENT_PREFIXES = ('', 'intro_', 'outro_')
SENTIMENT_DIMS = ('sentiment_valence', 'sentiment_arousal', 'sentiment_tension', 'sentiment_warmth')


def _series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors='coerce').fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _clip01(values: pd.Series | np.ndarray) -> pd.Series:
    return pd.Series(np.clip(values, 0.0, 1.0), index=getattr(values, 'index', None))


def _norm(values: pd.Series, vmin: float, vmax: float) -> pd.Series:
    if vmax <= vmin:
        return pd.Series(0.0, index=values.index, dtype=float)
    return _clip01((values - vmin) / (vmax - vmin))


def add_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive compact sentiment features from existing audio descriptors."""
    df = df.copy()
    for prefix in SENTIMENT_PREFIXES:
        existing_columns = [f'{prefix}{dimension}' for dimension in SENTIMENT_DIMS]
        if all(column in df.columns for column in existing_columns):
            if not df[existing_columns].isna().any().any():
                continue
        energy = _series(df, f'{prefix}energy')
        valence = _series(df, f'{prefix}valence')
        danceability = _series(df, f'{prefix}danceability')
        acousticness = _series(df, f'{prefix}acousticness')
        speechiness = _series(df, f'{prefix}speechiness')
        harmonic_ratio = _series(df, f'{prefix}harmonic_ratio')
        onset_strength = _norm(_series(df, f'{prefix}onset_strength'), 0.0, 5.0)
        brightness = _norm(_series(df, f'{prefix}spectral_centroid'), 1000.0, 5000.0)
        flatness = _norm(_series(df, f'{prefix}spectral_flatness'), 0.0, 0.5)
        zcr = _norm(_series(df, f'{prefix}zcr'), 0.0, 0.2)
        tempo_norm = _norm(_series(df, f'{prefix}tempo'), 60.0, 180.0)
        mode = _series(df, f'{prefix}mode', default=1.0)
        mode_major = _clip01(mode)

        roughness = _clip01(0.45 * flatness + 0.35 * zcr + 0.20 * (1.0 - harmonic_ratio))
        sentiment_valence = _clip01(0.55 * valence + 0.20 * mode_major + 0.15 * brightness + 0.10 * (1.0 - roughness))
        sentiment_arousal = _clip01(0.35 * energy + 0.20 * danceability + 0.20 * onset_strength + 0.15 * tempo_norm + 0.10 * brightness)
        sentiment_tension = _clip01(0.35 * roughness + 0.25 * onset_strength + 0.20 * (1.0 - valence) + 0.20 * (1.0 - harmonic_ratio))
        sentiment_warmth = _clip01(0.35 * harmonic_ratio + 0.25 * acousticness + 0.25 * (1.0 - brightness) + 0.15 * (1.0 - speechiness))

        df[f'{prefix}sentiment_valence'] = sentiment_valence.astype(float)
        df[f'{prefix}sentiment_arousal'] = sentiment_arousal.astype(float)
        df[f'{prefix}sentiment_tension'] = sentiment_tension.astype(float)
        df[f'{prefix}sentiment_warmth'] = sentiment_warmth.astype(float)

    return df
