"""components/tabs/sunburst.py"""

import pandas as pd
import streamlit as st

from modules.visualization import sunburst_hierarchy


def render(tab, df: pd.DataFrame) -> None:
    with tab:
        st.plotly_chart(sunburst_hierarchy(df), width="stretch")