"""components/tabs/kalibrasi.py — Elbow, Silhouette, HDBSCAN advisor."""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import streamlit as st

from modules.clustering import elbow_analysis, silhouette_analysis, hdbscan_min_size_analysis
from modules.visualization import elbow_chart, silhouette_chart
from config.settings import COLOR_ACCENT, COLOR_TEXT_DIM


def render(tab, feature_matrix: sp.csr_matrix, n_rows: int) -> None:
    with tab:
        st.markdown("#### Kalibrasi — Optimal K & Min Cluster Size")
        sub_km, sub_hdb = st.tabs(["KMeans — Elbow & Silhouette", "HDBSCAN — Min Size Advisor"])

        # ── KMeans ──────────────────────────────────────────
        with sub_km:
            st.caption("Elbow = kurva melandai · Silhouette = skor tertinggi")
            ceil  = st.session_state.get("_k_max", 50)
            k_max = st.slider("K range max", 3, ceil, min(15, ceil), key="k_max_elbow")

            with st.spinner("Menghitung..."):
                e_data = elbow_analysis(feature_matrix, range(2, k_max + 1))
                s_data = silhouette_analysis(feature_matrix, range(2, k_max + 1))

            ec, sc = st.columns(2)
            with ec: st.plotly_chart(elbow_chart(e_data), width="stretch")
            with sc: st.plotly_chart(silhouette_chart(s_data), width="stretch")

            bs  = s_data["best_k"]
            be  = e_data["best_k"]
            bsc = max(s_data["score"]) if s_data["score"] else 0

            m1, m2, m3 = st.columns(3)
            m1.metric("Rekomendasi Silhouette", f"K = {bs}", f"score {bsc:.3f}")
            m2.metric("Rekomendasi Elbow",      f"K = {be}")
            m3.metric(
                "Konsensus" if bs == be else "Saran",
                f"K = {bs} ✓" if bs == be else f"K = {bs}–{be}",
                "Kedua metode setuju" if bs == be else "Coba rentang ini",
            )
            with st.expander("Cara membaca"):
                st.markdown("""
**Elbow** — cari titik kurva mulai melandai (siku). Setelah titik itu, tambah K tidak banyak membantu.

**Silhouette** — pilih K dengan skor tertinggi (mendekati 1 = cluster kompak & terpisah).

Jika keduanya tidak setuju → pilih nilai di antara keduanya, cek scatter plot.
                """)

        # ── HDBSCAN ─────────────────────────────────────────
        with sub_hdb:
            sqrt_n = int(np.sqrt(n_rows))
            st.info(
                f"Data: **{n_rows:,} baris** · √n = **{sqrt_n}** · "
                f"Saran awal: **{max(5, sqrt_n // 2)}–{sqrt_n * 2}**"
            )

            if st.button("Jalankan Sweep", key="run_sweep", type="primary"):
                with st.spinner("Sweep HDBSCAN... (30–60 detik)"):
                    sweep = hdbscan_min_size_analysis(feature_matrix)

                if "error" in sweep:
                    st.error("hdbscan tidak terinstall. Jalankan: `pip install hdbscan`")
                else:
                    best_mcs = sweep["best_min_size"]
                    br       = next((r for r in sweep["results"] if r["min_size"] == best_mcs), {})

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Rekomendasi Min Size", best_mcs)
                    m2.metric("√n (baseline)", sweep["sqrt_n"])
                    m3.metric(
                        "Silhouette @ best",
                        f"{br.get('silhouette', 0):.3f}",
                        f"{br.get('n_clusters', '?')} kluster · noise {br.get('noise_ratio', 0):.1%}",
                    )

                    res_df = pd.DataFrame(sweep["results"])
                    st.dataframe(
                        res_df.rename(columns={
                            "min_size": "Min Size", "n_clusters": "# Kluster",
                            "noise_ratio": "Noise %", "mean_cluster_size": "Rata Ukuran",
                            "silhouette": "Silhouette",
                        }).style
                        .format({"Noise %": "{:.1%}", "Rata Ukuran": "{:.0f}", "Silhouette": "{:.3f}"})
                        .highlight_max(subset=["Silhouette"], color="#1a3a1a")
                        .highlight_min(subset=["Noise %"],    color="#1a2a3a"),
                        hide_index=True, width="stretch",
                    )

                    with st.expander("Cara membaca tabel"):
                        st.markdown(f"""
**Min Size** — nilai `min_cluster_size` yang diuji.
**# Kluster** — jumlah kluster ditemukan. Terlalu sedikit = over-merge, terlalu banyak = over-split.
**Noise %** — titik tidak masuk kluster manapun. < 15% = sehat.
**Silhouette** — pilih min_size dengan skor tertinggi dan noise % rendah.
**√n = {sweep['sqrt_n']}** adalah baseline empiris yang sering bekerja baik.
                        """)
            else:
                st.markdown(
                    f'<div style="color:{COLOR_TEXT_DIM};font-size:12px;padding:20px 0;">'
                    f'Tekan tombol untuk mulai sweep.<br>'
                    f'Baseline √n: <b style="color:{COLOR_ACCENT}">{sqrt_n}</b>'
                    f'</div>',
                    unsafe_allow_html=True,
                )