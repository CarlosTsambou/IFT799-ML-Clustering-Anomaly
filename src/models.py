# src/models.py

import time
import numpy as np

from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
)


# ============================
# 1. WRAPPER POUR MESURER TEMPS
# ============================

def timed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        return result, end - start
    return wrapper


# ============================
# 2. MÉTRIQUES
# ============================

def compute_internal_metrics(X, labels):
    """Métriques internes (non supervisées)."""
    # Certains algorithmes peuvent retourner -1 (bruit) → on gère les cas
    unique_labels = set(labels)
    if len(unique_labels) <= 1:
        return {
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski": np.nan
        }

    return {
        "silhouette": silhouette_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels),
        "calinski": calinski_harabasz_score(X, labels)
    }


def compute_external_metrics(true_labels, pred_labels):
    """Métriques externes (supervisées)."""
    return {
        "ARI": adjusted_rand_score(true_labels, pred_labels),
        "homogeneity": homogeneity_score(true_labels, pred_labels),
        "completeness": completeness_score(true_labels, pred_labels),
        "v_measure": v_measure_score(true_labels, pred_labels)
    }


# ============================
# 3. ALGOS DE CLUSTERING
# ============================

@timed
def run_kmeans(X, n_clusters, seed):
    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    labels = model.fit_predict(X)
    return labels


@timed
def run_dbscan(X, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return labels


@timed
def run_spectral(X, n_clusters, seed):
    model = SpectralClustering(
        n_clusters=n_clusters,
        random_state=seed,
        affinity="nearest_neighbors",
        n_neighbors=15
    )
    labels = model.fit_predict(X)
    return labels
