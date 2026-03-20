import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_mood_similarity(X):
    """Compute pairwise cosine similarity for feature matrix X."""
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 1.0)
    return sim


def combine_similarities(mood_sim, key_sim, mood_weight=2.0, key_weight=1.0):
    """Combine mood and key similarities with given weights (normalized)."""
    sim_mat = (mood_weight * mood_sim + key_weight * key_sim) / (mood_weight + key_weight)
    # symmetrize and clip
    sim_mat = (sim_mat + sim_mat.T) / 2
    sim_mat = np.clip(sim_mat, 0.0, 1.0)
    return sim_mat


def sparsify_knn(sim_mat, k=20):
    """Return a list of edges (i, j, weight) keeping top-k neighbors per node."""
    n = sim_mat.shape[0]
    edges = []
    k = min(k, n-1)
    for i in range(n):
        inds = np.argpartition(-sim_mat[i], kth=k)[:k+1]
        for j in inds:
            if i >= j:
                continue
            edges.append((int(i), int(j), float(sim_mat[i, j])))
    return edges
