"""components/tabs/treemap.py"""

import pandas as pd
import streamlit as st

from modules.visualization import treemap_nilai
from config.settings import DIM_OPTIONS, TREEMAP_GROUP_OPTIONS


def render(tab, df: pd.DataFrame) -> None:
    with tab:
        group_by = st.selectbox(
            "Group by", TREEMAP_GROUP_OPTIONS,
            format_func=lambda x: DIM_OPTIONS.get(x, x),
            key="treemap_by",
        )
        st.plotly_chart(treemap_nilai(df, group_col=group_by), width="stretch")