"""
components/metrics.py — Top metric cards row.
"""

import pandas as pd
import streamlit as st


def _fmt(v: float) -> str:
    if v >= 1e12: return f"Rp {v/1e12:.2f} T"
    if v >= 1e9:  return f"Rp {v/1e9:.1f} M"
    if v >= 1e6:  return f"Rp {v/1e6:.0f} jt"
    return f"Rp {v:,.0f}"


def render_metrics(df: pd.DataFrame, active_dims: list[str],
                   tfidf_weight: float, tag_weight: float,
                   clustering_dims: list[str]) -> float:
    """
    Render 4 metric cards. Returns total_nilai for use by legend.
    """
    total_paket = len(df)
    total_nilai = df["_nilai"].sum()
    n_kluster   = df["_cluster_label"].nunique()
    n_groups    = (
        df[active_dims].astype(str).agg(" · ".join, axis=1).nunique()
        if active_dims else 1
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Paket",      f"{total_paket:,}")
    c2.metric("Jumlah Kluster",   n_kluster)
    c3.metric("Jumlah Grup Aktif", n_groups)
    c4.metric("Total Nilai",      _fmt(total_nilai))

    if clustering_dims:
        st.info(
            f"**Fitur aktif:** {', '.join(clustering_dims)} · "
            f"Bobot teks {tfidf_weight:.0%} / tag {tag_weight:.0%}"
        )

    return total_nilai