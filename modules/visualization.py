"""
visualization.py — Plotly chart builders.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from config.settings import (
    COLOR_BG, COLOR_CARD_BG, COLOR_BORDER, COLOR_TEXT, COLOR_ACCENT, COLOR_CYAN,
    CHART_PALETTE, SCATTER_HOVER_COLS,
)

PALETTE = getattr(px.colors.qualitative, CHART_PALETTE)


def _colors(n: int) -> list[str]:
    return (PALETTE * ((n // len(PALETTE)) + 1))[:n]


def _dark_layout(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        template     = "plotly_dark",
        paper_bgcolor= COLOR_BG,
        plot_bgcolor = COLOR_CARD_BG,
        font         = dict(family="DM Sans, sans-serif", color=COLOR_TEXT),
        title        = dict(text=title, font=dict(size=14, color=COLOR_ACCENT)),
        margin       = dict(l=40, r=20, t=50, b=40),
        legend       = dict(
            font=dict(size=10, color=COLOR_TEXT),
            bgcolor=f"rgba(15,15,26,0.8)",
            bordercolor=COLOR_BORDER, borderwidth=1,
        ),
    )
    return fig


def scatter_cluster(df: pd.DataFrame, x_col: str = "_x", y_col: str = "_y",
                    color_col: str = "_cluster_label") -> go.Figure:
    hover_cols = [c for c in SCATTER_HOVER_COLS if c in df.columns]
    fig = px.scatter(
        df, x=x_col, y=y_col, color=color_col,
        color_discrete_sequence=_colors(df[color_col].nunique()),
        hover_data={x_col: False, y_col: False, color_col: True,
                    **{c: True for c in hover_cols}},
        custom_data=["_nilai"],
    )
    fig.update_traces(marker=dict(size=7, line=dict(width=0.5, color=COLOR_BORDER)))
    fig = _dark_layout(fig, "Scatter Plot — Kluster Nama Paket")
    fig.update_xaxes(showgrid=False, zeroline=False, title="")
    fig.update_yaxes(showgrid=False, zeroline=False, title="")
    return fig


def bar_distribution(df: pd.DataFrame, dim_col: str, title: str = "") -> go.Figure:
    counts = df[dim_col].value_counts().reset_index()
    counts.columns = [dim_col, "Jumlah"]
    fig = px.bar(counts, x=dim_col, y="Jumlah", color=dim_col,
                 color_discrete_sequence=_colors(counts[dim_col].nunique()), text="Jumlah")
    fig.update_traces(textposition="outside", textfont_size=10)
    fig = _dark_layout(fig, title or f"Distribusi — {dim_col}")
    fig.update_xaxes(title="", tickangle=-30)
    fig.update_yaxes(title="Jumlah Paket")
    return fig


def treemap_nilai(df: pd.DataFrame, group_col: str = "_cluster_label") -> go.Figure:
    agg = df.groupby(group_col)["_nilai"].sum().reset_index().rename(columns={"_nilai": "Total Nilai"})
    fig = px.treemap(agg, path=[group_col], values="Total Nilai", color="Total Nilai",
                     color_continuous_scale=[COLOR_CARD_BG, COLOR_ACCENT])
    return _dark_layout(fig, "Treemap — Total Nilai per Kluster")


def sunburst_hierarchy(df: pd.DataFrame) -> go.Figure:
    path_cols = [c for c in ["Cara Pengadaan", "Jenis Pengadaan", "Metode Pengadaan"] if c in df.columns]
    if not path_cols:
        return _dark_layout(go.Figure(), "Sunburst — Tidak ada data hierarki")
    fig = px.sunburst(df, path=path_cols, values="_nilai", color="_nilai",
                      color_continuous_scale=[COLOR_CARD_BG, COLOR_CYAN, COLOR_ACCENT])
    return _dark_layout(fig, "Sunburst — Cara → Jenis → Metode")


def elbow_chart(elbow_data: dict) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=elbow_data["k"], y=elbow_data["inertia"],
        mode="lines+markers",
        marker=dict(size=8, color=COLOR_ACCENT),
        line=dict(color=COLOR_ACCENT, width=2),
        name="Inertia",
    ))
    # Mark best_k
    if "best_k" in elbow_data and elbow_data["best_k"] in elbow_data["k"]:
        idx = elbow_data["k"].index(elbow_data["best_k"])
        fig.add_vline(x=elbow_data["best_k"], line_dash="dash",
                      line_color=COLOR_CYAN, opacity=0.6)
    fig = _dark_layout(fig, "Elbow Method — Inertia vs K")
    fig.update_xaxes(title="Jumlah Kluster (K)", dtick=1)
    fig.update_yaxes(title="Inertia")
    return fig


def silhouette_chart(sil_data: dict) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=sil_data["k"], y=sil_data["score"],
        mode="lines+markers",
        marker=dict(size=8, color=COLOR_CYAN),
        line=dict(color=COLOR_CYAN, width=2),
        name="Silhouette",
    ))
    if "best_k" in sil_data and sil_data["best_k"] in sil_data["k"]:
        fig.add_vline(x=sil_data["best_k"], line_dash="dash",
                      line_color=COLOR_ACCENT, opacity=0.6)
    fig = _dark_layout(fig, "Silhouette Score vs K")
    fig.update_xaxes(title="Jumlah Kluster (K)", dtick=1)
    fig.update_yaxes(title="Silhouette Score")
    return fig


def cluster_stats_bar(stats_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        stats_df, y="_cluster_label", x="total_nilai", orientation="h",
        color="_cluster_label",
        color_discrete_sequence=_colors(len(stats_df)),
        text=stats_df["total_nilai"].apply(lambda v: f"Rp {v:,.0f}"),
    )
    fig.update_traces(textposition="outside", textfont_size=9)
    fig = _dark_layout(fig, "Total Nilai per Kluster")
    fig.update_yaxes(title="", categoryorder="total ascending")
    fig.update_xaxes(title="Total Nilai (Rp)")
    return fig