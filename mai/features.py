import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA


def add_log_tempo(df, tempo_col='Tempo', out_col='log_tempo'):
    """Add a log2-tempo feature (octave aware)."""
    eps = 1e-6
    if tempo_col in df.columns:
        df[out_col] = np.log2(df[tempo_col].fillna(df[tempo_col].median()) + eps)
    else:
        df[out_col] = 0.0
    return df


def scale_and_pca(X_raw, do_pca=True, pca_variance=0.95):
    """Apply RobustScaler and optional PCA. Returns transformed array and fitted objects."""
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_raw)
    if do_pca and X_scaled.shape[1] > 1:
        pca = PCA(n_components=pca_variance, svd_solver='full')
        X = pca.fit_transform(X_scaled)
    else:
        pca = None
        X = X_scaled
    return X, scaler, pca
