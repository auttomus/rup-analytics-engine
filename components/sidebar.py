"""
components/sidebar.py — Sidebar UI.
render_sidebar() → returns all params needed by pipeline + app.
"""

import os
import streamlit as st

from modules.data_loader import discover_csv_files
from modules.reduction import METHODS as REDUCTION_METHODS
from config.settings import (
    DIM_OPTIONS, CLUSTER_DIM_OPTIONS, DIM_COLOR_DEFAULT,
    KMEANS_DEFAULT_K, HDBSCAN_DEFAULT_MIN_SIZE,
    TAG_WEIGHT_DEFAULT, TAG_WEIGHT_MIN, TAG_WEIGHT_MAX, TAG_WEIGHT_STEP,
    TEXT_BACKEND_LABEL,
    COLOR_ACCENT, COLOR_TEXT_DIM,
)


def render_sidebar(base_dir: str) -> dict:
    """
    Render full sidebar. Returns dict of all user-selected params.

    Returns:
        {
            csv_path, k_clusters, reduction_method, algo, hdbscan_min,
            active_dims, clustering_dims, tag_weight, tfidf_weight,
        }
    """
    with st.sidebar:
        st.markdown('<div class="logo-text">RUP · ENGINE</div>', unsafe_allow_html=True)
        st.markdown('<div class="logo-sub">Spatial Analytics Engine · 2026</div>', unsafe_allow_html=True)
        st.markdown("---")

        # ── Data source ─────────────────────────────────────
        st.markdown("##### Sumber Data")
        csv_files   = discover_csv_files(base_dir)
        source_mode = st.radio("Mode", ["File Lokal", "Upload CSV"],
                               horizontal=True, label_visibility="collapsed")

        csv_path = None
        if source_mode == "File Lokal":
            if csv_files:
                csv_names = [os.path.basename(f) for f in csv_files]
                chosen    = st.selectbox("Pilih File CSV", csv_names, index=0)
                csv_path  = csv_files[csv_names.index(chosen)]
            else:
                st.warning("Tidak ada file CSV di direktori.")
        else:
            uploaded = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded:
                tmp_path = os.path.join(base_dir, f"_uploaded_{uploaded.name}")
                with open(tmp_path, "wb") as f:
                    f.write(uploaded.getbuffer())
                csv_path = tmp_path

        st.markdown("---")

        # ── Clustering params ────────────────────────────────
        st.markdown("##### Parameter Clustering")
        reduction_method = st.selectbox(
            "Metode Reduksi 2D",
            REDUCTION_METHODS,
            index=REDUCTION_METHODS.index("t-SNE") if "t-SNE" in REDUCTION_METHODS else 0,
        )
        algo = st.radio("Algoritma", ["KMeans", "HDBSCAN"], horizontal=True, index=1)
        k_max = st.session_state.get("_k_max", 50)
        k_clusters = min(KMEANS_DEFAULT_K, k_max)
        hdbscan_min = HDBSCAN_DEFAULT_MIN_SIZE
        if algo == "KMeans":
            k_clusters = st.slider(
                "Jumlah Kluster (K)",
                min_value=2,
                max_value=k_max,
                value=min(KMEANS_DEFAULT_K, k_max),
                key="kmeans_k_slider",
            )
        else:
            hdbscan_min = st.slider(
                "Min Cluster Size",
                min_value=5,
                max_value=100,
                value=HDBSCAN_DEFAULT_MIN_SIZE,
                key="hdbscan_min_slider",
            )
        st.caption(f"Backend teks: {TEXT_BACKEND_LABEL}")

        st.markdown("---")

        # ── Active dims (coloring) ───────────────────────────
        st.markdown("##### Dimensi Aktif")
        st.caption("Coloring scatter plot")
        active_dims = [
            k for k, v in DIM_OPTIONS.items()
            if st.checkbox(v, value=(k == DIM_COLOR_DEFAULT), key=f"dim_{k}")
        ]

        st.markdown("---")

        # ── Clustering features (tag matrix) ─────────────────
        st.markdown("##### Fitur Clustering")
        st.caption("Pengaruhi posisi kluster")
        clustering_dims = [
            k for k, v in CLUSTER_DIM_OPTIONS.items()
            if st.checkbox(v, value=False, key=f"clust_{k}")
        ]

        tag_weight   = TAG_WEIGHT_DEFAULT
        tfidf_weight = 1.0
        if clustering_dims:
            tag_weight = st.slider(
                "Bobot Tag vs Teks",
                TAG_WEIGHT_MIN, TAG_WEIGHT_MAX, TAG_WEIGHT_DEFAULT, TAG_WEIGHT_STEP,
                help="0.1 = teks dominan · 0.9 = tag dominan",
            )
            tfidf_weight = 1.0 - tag_weight
            st.caption(f"Teks: {tfidf_weight:.0%} · Tag: {tag_weight:.0%}")

        st.markdown("---")

    return {
        "csv_path":        csv_path,
        "k_clusters":      k_clusters,
        "reduction_method": reduction_method,
        "algo":            algo,
        "hdbscan_min":     hdbscan_min,
        "active_dims":     active_dims,
        "clustering_dims": clustering_dims,
        "tag_weight":      tag_weight,
        "tfidf_weight":    tfidf_weight,
    }