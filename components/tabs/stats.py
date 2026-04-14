"""components/tabs/stats.py"""

import pandas as pd
import streamlit as st

from modules.clustering import compute_cluster_stats
from modules.visualization import cluster_stats_bar
from config.settings import CLUSTER_DETAIL_COLS


def render(tab, df: pd.DataFrame) -> None:
    with tab:
        st.markdown("#### Statistik per Kluster")
        stats_df = compute_cluster_stats(df)
        st.plotly_chart(cluster_stats_bar(stats_df), width="stretch")

        st.markdown("##### Detail Kluster")
        for _, row in stats_df.iterrows():
            with st.expander(f"{row['_cluster_label']} — {row['jumlah_paket']} paket"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Nilai", f"Rp {row['total_nilai']:,.0f}")
                c2.metric("Rata-rata",   f"Rp {row['rata_nilai']:,.0f}")
                c3.metric("Rentang",     f"Rp {row['min_nilai']:,.0f} — Rp {row['max_nilai']:,.0f}")

                cdf   = df[df["_cluster_label"] == row["_cluster_label"]]
                scols = [c for c in CLUSTER_DETAIL_COLS if c in cdf.columns]
                st.dataframe(
                    cdf[scols].sort_values("_nilai", ascending=False),
                    width="stretch", hide_index=True,
                )