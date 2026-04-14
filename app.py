"""
app.py — RUP Spatial Analytics Engine
Orchestrator only. No business logic here.
"""

import os
import streamlit as st
import scipy.sparse as sp

from modules.data_loader import load_and_clean
from pipeline import run_pipeline
from components.sidebar import render_sidebar
from components.metrics import render_metrics
from components.scatter import render_scatter
from components.legend import render_legend
from components.tabs import distribusi, stats, treemap, sunburst, kalibrasi, data
from config.settings import (
    K_MAX_ROW_DIVISOR, K_MAX_HARD,
    COLOR_BG, COLOR_SIDEBAR, COLOR_BORDER, COLOR_CARD_BG, COLOR_CARD_BG2,
    COLOR_TEXT, COLOR_TEXT_DIM, COLOR_ACCENT, COLOR_CYAN, COLOR_HEADER,
)

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="RUP · Spatial Analytics Engine",
    page_icon="", layout="wide", initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
    .stApp {{ background-color: {COLOR_BG}; }}
    section[data-testid="stSidebar"] {{ background-color: {COLOR_SIDEBAR}; border-right: 1px solid {COLOR_BORDER}; }}
    div[data-testid="stMetric"] {{ background: linear-gradient(135deg, {COLOR_CARD_BG} 0%, {COLOR_CARD_BG2} 100%); border: 1px solid {COLOR_BORDER}; border-radius: 8px; padding: 12px 16px; }}
    div[data-testid="stMetric"] label {{ color: {COLOR_TEXT_DIM} !important; font-size: 11px !important; letter-spacing: 2px; text-transform: uppercase; }}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{ color: {COLOR_CYAN} !important; font-weight: 700 !important; }}
    button[data-baseweb="tab"] {{ color: {COLOR_TEXT_DIM} !important; font-size: 12px !important; letter-spacing: 1px; }}
    button[data-baseweb="tab"][aria-selected="true"] {{ color: {COLOR_ACCENT} !important; border-bottom-color: {COLOR_ACCENT} !important; }}
    h1, h2, h3 {{ color: {COLOR_HEADER} !important; }}
    .logo-text {{ font-family: 'Space Mono', monospace; font-size: 11px; letter-spacing: 5px; color: {COLOR_ACCENT}; margin-bottom: 2px; }}
    .logo-sub {{ font-family: 'Space Mono', monospace; font-size: 9px; color: {COLOR_TEXT_DIM}; letter-spacing: 1px; }}
    hr {{ border-color: {COLOR_BORDER} !important; }}
    .stDataFrame {{ border: 1px solid {COLOR_BORDER}; border-radius: 4px; }}
    .streamlit-expanderHeader {{ color: {COLOR_TEXT} !important; font-size: 12px !important; }}
    .cluster-card {{ background: linear-gradient(135deg, {COLOR_CARD_BG}, {COLOR_CARD_BG2}); border: 1px solid {COLOR_BORDER}; border-radius: 6px; padding: 10px 14px; margin-bottom: 6px; }}
    .cluster-card .cc-label {{ font-family: monospace; font-size: 10px; color: {COLOR_ACCENT}; margin-bottom: 4px; }}
    .cluster-card .cc-val {{ font-family: monospace; font-size: 9px; color: {COLOR_TEXT}; }}
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ─────────────────────────────────────────────────
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
params   = render_sidebar(data_dir)

if not params["csv_path"]:
    st.info("Pilih atau upload file CSV di sidebar untuk memulai.")
    st.stop()

# ── ADAPTIVE K MAX ───────────────────────────────────────────
try:
    _df_meta = load_and_clean(params["csv_path"])
except ValueError as e:
    st.error(str(e)); st.stop()

st.session_state["_k_max"] = max(2, min(
    _df_meta["Nama Paket"].nunique(),
    len(_df_meta) // K_MAX_ROW_DIVISOR,
    K_MAX_HARD,
))

# ── PIPELINE ─────────────────────────────────────────────────
with st.spinner("Clustering... "):
    try:
        df, feature_dense, _ = run_pipeline(
            csv_path         = params["csv_path"],
            k_clusters       = params["k_clusters"],
            reduction_method = params["reduction_method"],
            algo             = params["algo"],
            clustering_dims  = tuple(sorted(params["clustering_dims"])),
            tag_weight       = round(params["tag_weight"], 2),
            hdbscan_min      = params["hdbscan_min"] if params["algo"] == "HDBSCAN" else 0,
        )
    except Exception as e:
        st.error(f"Pipeline error: {e}"); st.stop()

feature_matrix = sp.csr_matrix(feature_dense)

# ── METRICS ─────────────────────────────────────────────────
total_nilai = render_metrics(
    df,
    active_dims     = params["active_dims"],
    tfidf_weight    = params["tfidf_weight"],
    tag_weight      = params["tag_weight"],
    clustering_dims = params["clustering_dims"],
)

# ── SCATTER ─────────────────────────────────────────────────
render_scatter(df, active_dims=params["active_dims"])

# ── TABS ─────────────────────────────────────────────────────
tab_dist, tab_stats, tab_tree, tab_sun, tab_kal, tab_dat = st.tabs([
    "Distribusi", "Statistik Kluster", "Treemap", "Sunburst", "Kalibrasi", "Data",
])

distribusi.render(tab_dist, df)
stats.render(tab_stats, df)
treemap.render(tab_tree, df)
sunburst.render(tab_sun, df)
kalibrasi.render(tab_kal, feature_matrix, n_rows=len(df))
data.render(tab_dat, df)

# ── LEGEND ──────────────────────────────────────────────────
render_legend(df, total_nilai)