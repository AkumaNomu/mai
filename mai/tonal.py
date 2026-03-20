import numpy as np


def kk_reference_profiles():
    """Return normalized Krumhansl-Kessler profiles for major/minor modes."""
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=float)
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=float)
    major_profile = major_profile / np.linalg.norm(major_profile)
    minor_profile = minor_profile / np.linalg.norm(minor_profile)
    return major_profile, minor_profile


def kk_key_profiles(keys, modes):
    """Build KK key profiles from arrays of key indexes and modes."""
    major_profile, minor_profile = kk_reference_profiles()
    n = len(keys)
    profiles = np.zeros((n, 12), dtype=float)
    for i in range(n):
        key = int(keys[i]) % 12
        base = major_profile if int(modes[i]) == 1 else minor_profile
        profiles[i] = np.roll(base, -key)
    return profiles


def kk_profile_similarity(from_profiles, to_profiles=None):
    """Compute pairwise cosine similarity between one or two profile matrices."""
    if to_profiles is None:
        to_profiles = from_profiles
    num = from_profiles @ to_profiles.T
    den = np.linalg.norm(from_profiles, axis=1)[:, None] * np.linalg.norm(to_profiles, axis=1)[None, :]
    sim = np.divide(num, den, out=np.zeros_like(num), where=den != 0)
    return np.clip(sim, 0.0, 1.0)


def kk_key_similarity(df, key_col='Key', mode_col='Mode'):
    """Compute key compatibility matrix using Krumhansl-Kessler profiles.

    Returns an (n,n) matrix with values in [0,1].
    """
    n = len(df)
    if key_col in df.columns:
        keys = df[key_col].fillna(0).astype(int).to_numpy()
    else:
        keys = np.zeros(n, dtype=int)
    modes = df[mode_col].fillna(1).astype(int).to_numpy() if mode_col in df.columns else np.ones(n, dtype=int)
    song_profiles = kk_key_profiles(keys, modes)
    return kk_profile_similarity(song_profiles)


def kk_key_transition_similarity(df, from_key_col='key', from_mode_col='mode', to_key_col='key', to_mode_col='mode'):
    """Compute directed key compatibility from source columns to target columns."""
    n = len(df)
    from_keys = df[from_key_col].fillna(0).astype(int).to_numpy() if from_key_col in df.columns else np.zeros(n, dtype=int)
    from_modes = df[from_mode_col].fillna(1).astype(int).to_numpy() if from_mode_col in df.columns else np.ones(n, dtype=int)
    to_keys = df[to_key_col].fillna(0).astype(int).to_numpy() if to_key_col in df.columns else np.zeros(n, dtype=int)
    to_modes = df[to_mode_col].fillna(1).astype(int).to_numpy() if to_mode_col in df.columns else np.ones(n, dtype=int)
    from_profiles = kk_key_profiles(from_keys, from_modes)
    to_profiles = kk_key_profiles(to_keys, to_modes)
    return kk_profile_similarity(from_profiles, to_profiles)
