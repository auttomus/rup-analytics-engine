"""
pipeline.py — Cached clustering pipeline.
Single entry point for all heavy computation.
Re-runs only when parameters change.
"""

import streamlit as st

from modules.data_loader import load_and_clean
from modules.clustering import (
    build_text_embeddings, build_tag_matrix, combine_matrices,
    run_kmeans, run_hdbscan, extract_cluster_labels,
)
from modules.reduction import reduce_2d
from config.settings import TEXT_EMBED_MODEL, TEXT_EMBED_DEVICE, TEXT_EMBED_BATCH_SIZE


@st.cache_data(show_spinner=False)
def run_pipeline(
    csv_path: str,
    k_clusters: int,
    reduction_method: str,
    algo: str,
    clustering_dims: tuple,   # tuple → hashable cache key
    tag_weight: float,
    hdbscan_min: int,
) -> tuple:
    """
    Returns: (df_enriched, feature_matrix_dense, processed_texts)

    feature_matrix returned as dense numpy — sparse not serializable by st.cache_data.
    Caller converts back: scipy.sparse.csr_matrix(feature_dense)
    """
    df                      = load_and_clean(csv_path)
    text_mat, _, proc_texts = build_text_embeddings(
        df["Nama Paket"].tolist(),
        model_name=TEXT_EMBED_MODEL,
        device=TEXT_EMBED_DEVICE,
        batch_size=TEXT_EMBED_BATCH_SIZE,
    )
    tag_mat                 = build_tag_matrix(df, list(clustering_dims))
    combined                = combine_matrices(text_mat, tag_mat, 1.0 - tag_weight, tag_weight)

    if algo == "HDBSCAN":
        try:
            labels, model = run_hdbscan(combined, min_cluster_size=hdbscan_min)
        except ImportError:
            st.warning("hdbscan tidak terinstall → fallback KMeans")
            labels, model = run_kmeans(combined, k_clusters)
    else:
        labels, model = run_kmeans(combined, k_clusters)

    cl, ex = extract_cluster_labels(
        model,
        None,
        texts=df["Nama Paket"].astype(str).tolist(),
        labels=labels,
        feature_matrix=combined,
    )
    
    df["_cluster_id"]      = labels
    df["_cluster_label"]   = df["_cluster_id"].apply(lambda x: f"K{x}: {cl.get(int(x), 'Lainnya')}")
    df["_cluster_example"] = df["_cluster_id"].apply(lambda x: ex.get(int(x), "-"))

    coords   = reduce_2d(combined, method=reduction_method)
    df["_x"] = coords[:, 0]
    df["_y"] = coords[:, 1]

    return df, combined.toarray(), proc_texts