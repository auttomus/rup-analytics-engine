"""
data_loader.py — CSV loading, cleaning, value binning.
"""

import pandas as pd
import streamlit as st
import glob
import os

from config.settings import (
    REQUIRED_COLS, CATEGORICAL_COLS, OPTIONAL_COLS, NILAI_BINS,
)


def bin_nilai(v: float) -> str:
    for threshold, label in NILAI_BINS:
        if v < threshold:
            return label
    return NILAI_BINS[-1][1]


def discover_csv_files(directory: str) -> list[str]:
    return sorted(glob.glob(os.path.join(directory, "*.csv")))


@st.cache_data(show_spinner=False)
def load_and_clean(filepath: str) -> pd.DataFrame:
    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            df = pd.read_csv(filepath, encoding=enc)
            break
        except (UnicodeDecodeError, Exception):
            continue
    else:
        raise ValueError(f"Gagal membaca CSV: encoding tidak dikenali — {filepath}")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom tidak ditemukan: {', '.join(missing)}")

    df["Nama Paket"] = df["Nama Paket"].fillna("TANPA NAMA").astype(str)
    df["Total Nilai (Rp)"] = (
        df["Total Nilai (Rp)"]
        .astype(str)
        .str.replace(r"[^0-9.]", "", regex=True)
        .replace("", "0")
        .astype(float)
    )
    df["Kode RUP"] = pd.to_numeric(df["Kode RUP"], errors="coerce").fillna(0.0).astype(float)

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("TIDAK TERDEFINISI").astype(str)

    for col in OPTIONAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("TIDAK TERDEFINISI").astype(str)

    df["_nilai"]     = df["Total Nilai (Rp)"]
    df["_nilai_bin"] = df["_nilai"].apply(bin_nilai)
    return df