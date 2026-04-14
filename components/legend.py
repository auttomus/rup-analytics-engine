"""
components/legend.py — Sidebar cluster legend with search.
"""

import pandas as pd
import streamlit as st

from modules.clustering import compute_cluster_stats
from config.settings import COLOR_ACCENT, COLOR_TEXT_DIM


def render_legend(df: pd.DataFrame, total_nilai: float) -> None:
    """Render searchable cluster legend cards in sidebar."""
    with st.sidebar:
        st.markdown("##### Legenda Kluster")
        q = st.text_input(
            "Cari kluster...", placeholder="Ketik nama/kata kunci",
            label_visibility="collapsed", key="legend_search",
        ).strip().lower()

        stats = compute_cluster_stats(df)
        filtered = (
            stats[stats["_cluster_label"].str.lower().str.contains(q, na=False)]
            if q else stats
        )

        if filtered.empty:
            st.caption("Tidak ada kluster yang cocok.")
            return

        for i, row in filtered.iterrows():
            pct = row["total_nilai"] / total_nilai * 100 if total_nilai > 0 else 0
            st.markdown(
                f'<div class="cluster-card">'
                f'<div class="cc-label">K{i}: {row["_cluster_label"].split(": ", 1)[-1]}</div>'
                f'<div class="cc-val">{row["jumlah_paket"]} paket · '
                f'Rp {row["total_nilai"]:,.0f} ({pct:.1f}%)</div>'
                f'</div>',
                unsafe_allow_html=True,
            )