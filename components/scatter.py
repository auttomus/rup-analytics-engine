"""
components/scatter.py — Scatter plot with color dimension logic.
"""

import pandas as pd
import streamlit as st

from modules.visualization import scatter_cluster


def render_scatter(df: pd.DataFrame, active_dims: list[str]) -> str:
    """
    Resolve color_col from active_dims, render scatter. Returns color_col used.
    """
    if active_dims:
        if len(active_dims) == 1:
            color_col = active_dims[0]
        else:
            df["_composite_dim"] = df[active_dims].astype(str).agg(" · ".join, axis=1)
            color_col = "_composite_dim"
    else:
        color_col = "_cluster_label"

    st.plotly_chart(
        scatter_cluster(df, color_col=color_col),
        width="stretch",
        key="main_scatter",
    )
    return color_col