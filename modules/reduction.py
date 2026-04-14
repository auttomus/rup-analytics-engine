"""
reduction.py — PCA, t-SNE, UMAP dimensionality reduction.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from config.settings import (
    REDUCTION_RANDOM_STATE,
    TSNE_PERPLEXITY_MIN, TSNE_PERPLEXITY_MAX, TSNE_PERPLEXITY_DIVISOR,
    UMAP_N_NEIGHBORS_MIN, UMAP_N_NEIGHBORS_MAX, UMAP_N_NEIGHBORS_DIVISOR,
)

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

METHODS = ["PCA", "t-SNE"] + (["UMAP"] if HAS_UMAP else [])


def reduce_2d(matrix, method: str = "PCA") -> np.ndarray:
    dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    n     = dense.shape[0]

    if method == "PCA":
        return PCA(n_components=2, random_state=REDUCTION_RANDOM_STATE).fit_transform(dense)

    if method == "t-SNE":
        perplexity = min(TSNE_PERPLEXITY_MAX, max(TSNE_PERPLEXITY_MIN, n // TSNE_PERPLEXITY_DIVISOR))
        return TSNE(
            n_components=2, perplexity=perplexity,
            random_state=REDUCTION_RANDOM_STATE,
            init="pca", learning_rate="auto",
        ).fit_transform(dense)

    if method == "UMAP" and HAS_UMAP:
        n_neighbors = min(UMAP_N_NEIGHBORS_MAX, max(UMAP_N_NEIGHBORS_MIN, n // UMAP_N_NEIGHBORS_DIVISOR))
        return UMAP(
            n_components=2, n_neighbors=n_neighbors,
            random_state=REDUCTION_RANDOM_STATE,
        ).fit_transform(dense)

    raise ValueError(f"Unknown reduction method: {method}")