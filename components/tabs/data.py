"""components/tabs/data.py"""

import pandas as pd
import streamlit as st

from config.settings import DATA_DISPLAY_COLS


def render(tab, df: pd.DataFrame) -> None:
    with tab:
        st.markdown("#### Eksplorasi Data")
        all_cls  = ["Semua"] + sorted(df["_cluster_label"].unique().tolist())
        f_cls    = st.selectbox("Filter Kluster", all_cls, key="data_filter")
        ddf      = df if f_cls == "Semua" else df[df["_cluster_label"] == f_cls]
        dcols    = [c for c in DATA_DISPLAY_COLS if c in ddf.columns]

        st.dataframe(
            ddf[dcols].sort_values("_nilai", ascending=False),
            width="stretch", hide_index=True, height=500,
        )
        st.download_button(
            "Download CSV",
            ddf[dcols].to_csv(index=False).encode("utf-8"),
            file_name="rup_clustered_export.csv",
            mime="text/csv",
        )