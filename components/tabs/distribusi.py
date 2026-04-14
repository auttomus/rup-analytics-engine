"""components/tabs/distribusi.py"""

import pandas as pd
import streamlit as st

from modules.visualization import bar_distribution
from config.settings import DIM_OPTIONS


def render(tab, df: pd.DataFrame) -> None:
    with tab:
        dim = st.selectbox("Dimensi", list(DIM_OPTIONS.keys()),
                           format_func=lambda x: DIM_OPTIONS.get(x, x), key="bar_dim")
        st.plotly_chart(
            bar_distribution(df, dim, title=f"Distribusi — {DIM_OPTIONS.get(dim, dim)}"),
            width="stretch",
        )