"""
config/settings.py — Single source of truth untuk semua konstanta RUP Analytics Engine.

Ubah di sini → berlaku di seluruh aplikasi.
Tidak ada magic number yang tersebar di file lain.
"""

# ══════════════════════════════════════════════════════════════
# DATA LOADER
# ══════════════════════════════════════════════════════════════

# Kolom wajib di CSV input
REQUIRED_COLS = [
    "Nama Paket",
    "Kode RUP",
    "Nama Instansi",
    "Nama Satuan Kerja",
    "Sumber Dana",
    "Tahun Anggaran",
    "Total Nilai (Rp)",
    "Cara Pengadaan",
    "Jenis Pengadaan",
    "Metode Pengadaan",
    "Produk Dalam Negeri",
]

# Kolom kategorikal yang di-fill NaN → "TIDAK TERDEFINISI"
CATEGORICAL_COLS = [
    "Nama Instansi",
    "Nama Satuan Kerja",
    "Sumber Dana",
    "Tahun Anggaran",
    "Cara Pengadaan",
    "Jenis Pengadaan",
    "Metode Pengadaan",
    "Produk Dalam Negeri",
]

# Kolom opsional (tidak wajib ada, tapi di-clean jika ada)
OPTIONAL_COLS = []

# Bins untuk pengelompokan nilai rupiah
# Format: (batas_atas_eksklusif, label)
NILAI_BINS = [
    (1e6, "< 1 jt"),
    (10e6, "1–10 jt"),
    (50e6, "10–50 jt"),
    (200e6, "50–200 jt"),
    (1e9, "200 jt–1 m"),
    (10e9, "1–10 m"),
    (100e9, "10–100 m"),
    (float("inf"), "> 100 m"),
]


# ══════════════════════════════════════════════════════════════
# TF-IDF
# ══════════════════════════════════════════════════════════════

# Maksimum jumlah fitur TF-IDF
TFIDF_MAX_FEATURES = 1000

# Ngram range: (1, 2) = unigram + bigram
TFIDF_NGRAM_RANGE = (1, 2)

# max_df: buang term yang muncul di lebih dari X% dokumen (noise)
# 0.40 = term di >40% dokumen = tidak diskriminatif
TFIDF_MAX_DF = 0.90

# min_df: dihitung dinamis = max(TFIDF_MIN_DF_FLOOR, n_docs * TFIDF_MIN_DF_RATIO)
TFIDF_MIN_DF_FLOOR = 2  # minimal 2 dokumen
TFIDF_MIN_DF_RATIO = 0.01  # atau 1% dari total dokumen

# Sublinear TF: log(1 + tf) — dampen frequent terms
TFIDF_SUBLINEAR_TF = True

# Token minimum length (karakter)
TFIDF_MIN_TOKEN_LEN = 2


# ══════════════════════════════════════════════════════════════
# TEXT EMBEDDING
# ══════════════════════════════════════════════════════════════

TEXT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_EMBED_DEVICE = "auto"  # auto | cpu | cuda
TEXT_EMBED_BATCH_SIZE = 64
TEXT_BACKEND_LABEL = "MiniLM (all-MiniLM-L6-v2)"


# ══════════════════════════════════════════════════════════════
# CLUSTER LABELING
# ══════════════════════════════════════════════════════════════

LABEL_EXEMPLAR_TOP_K = 3
LABEL_MAX_WORDS = 8
LABEL_MIN_CONFIDENCE = 0.35

# Konfigurasi Baru untuk Multi-Exemplar Token Scoring
LABEL_TOP_N_TERMS = 3
LABEL_MIN_TERM_SCORE = 0.1
LABEL_EXAMPLE_MAX_WORDS = 15
LABEL_NGRAM_RANGE = (1, 2)

LABEL_GENERIC_TERMS: set[str] = {
     # Generic Indonesian
    "dan",
    "atau",
    "yang",
    "dengan",
    "dalam",
    "dari",
    "pada",
    "ke",
    "di",
    "untuk",
    "oleh",
    "atas",
    "serta",
    "tidak",
    "secara",
    "lain",
    "ini",
    "itu",
    "juga",
    "akan",
    "telah",
    "dapat",
    "sudah",
    "ada",
    "hal",
    "agar",
    "yaitu",
    "adalah",
    "melalui",
    "sebagai",
    "sesuai",
    "antara",
    # Generic terms yang dapat digunakan bila ingin dihilangkan dari label
    "belanja", "pengadaan", "barang", "jasa", "kegiatan", "paket",
    # "biaya", "layanan", "pelaksanaan", "penyediaan", "administrasi",
    # "rapat", "koordinasi", "pertemuan", "konsultasi", "fasilitas",
}


# ══════════════════════════════════════════════════════════════
# CLUSTERING — KMeans
# ══════════════════════════════════════════════════════════════

# n_init untuk clustering utama (lebih tinggi = lebih stabil, lebih lambat)
KMEANS_N_INIT = 40

# n_init untuk analisis elbow/silhouette (tidak perlu se-stable clustering utama)
KMEANS_N_INIT_ANALYSIS = 15

KMEANS_MAX_ITER = 500
KMEANS_MAX_ITER_ANALYSIS = 300
KMEANS_RANDOM_STATE = 42

# Default K saat pertama load
KMEANS_DEFAULT_K = 20

# Batas atas K adaptif: min(n_unique_names, n_rows // K_MAX_ROW_DIVISOR, K_MAX_HARD)
K_MAX_ROW_DIVISOR = 2
K_MAX_HARD = 100


# ══════════════════════════════════════════════════════════════
# CLUSTERING — HDBSCAN
# ══════════════════════════════════════════════════════════════

HDBSCAN_DEFAULT_MIN_SIZE = 5
HDBSCAN_MIN_SAMPLES_RATIO = 3  # min_samples = max(1, min_size // ratio)
HDBSCAN_METRIC = "euclidean"
HDBSCAN_CLUSTER_SELECTION = "eom"

# Sweep range: dari HDBSCAN_SWEEP_MIN sampai n_samples * HDBSCAN_SWEEP_MAX_RATIO
HDBSCAN_SWEEP_MIN = 5
HDBSCAN_SWEEP_MAX_RATIO = 0.05  # max = 5% dari total data
HDBSCAN_SWEEP_FIXED = [5, 8, 10, 15, 20, 30, 50]
HDBSCAN_SWEEP_N_LOGSPACE = 12

# Noise ratio threshold: sweep result dianggap "viable" jika noise < threshold
HDBSCAN_NOISE_THRESHOLD = 0.15


# ══════════════════════════════════════════════════════════════
# SILHOUETTE ANALYSIS
# ══════════════════════════════════════════════════════════════

# Sampel maksimum untuk silhouette (O(n²) tanpa sampling)
SILHOUETTE_SAMPLE_SIZE = 2000
SILHOUETTE_HDBSCAN_SAMPLE_SIZE = 1500
SILHOUETTE_RANDOM_STATE = 42


# ══════════════════════════════════════════════════════════════
# TAG MATRIX
# ══════════════════════════════════════════════════════════════

# Default bobot tag vs tfidf (bisa diubah user lewat slider)
TAG_WEIGHT_DEFAULT = 0.5
TAG_WEIGHT_MIN = 0.1
TAG_WEIGHT_MAX = 0.9
TAG_WEIGHT_STEP = 0.05


# ══════════════════════════════════════════════════════════════
# DIMENSIONALITY REDUCTION
# ══════════════════════════════════════════════════════════════

REDUCTION_RANDOM_STATE = 42

# t-SNE: perplexity = min(MAX, max(MIN, n_samples // DIVISOR))
TSNE_PERPLEXITY_MIN = 5
TSNE_PERPLEXITY_MAX = 30
TSNE_PERPLEXITY_DIVISOR = 4

# UMAP: n_neighbors = min(MAX, max(MIN, n_samples // DIVISOR))
UMAP_N_NEIGHBORS_MIN = 3
UMAP_N_NEIGHBORS_MAX = 15
UMAP_N_NEIGHBORS_DIVISOR = 5


# ══════════════════════════════════════════════════════════════
# UI — DIMENSI
# ══════════════════════════════════════════════════════════════

# Dimensi yang bisa dipakai untuk coloring scatter plot
# key = nama kolom di df, value = label tampil di UI
DIM_OPTIONS: dict[str, str] = {
    "Nama Instansi": "Nama Instansi",
    "Nama Satuan Kerja": "Nama Satuan Kerja",
    "Sumber Dana": "Sumber Dana",
    "Tahun Anggaran": "Tahun Anggaran",
    "Cara Pengadaan": "Cara Pengadaan",
    "Jenis Pengadaan": "Jenis Pengadaan",
    "Metode Pengadaan": "Metode Pengadaan",
    "Produk Dalam Negeri": "Produk DN",
    "_nilai_bin": "Kelompok Nilai",
    "_cluster_label": "Kluster Nama Paket",
}

# Dimensi yang bisa dipakai sebagai fitur clustering (tag matrix)
CLUSTER_DIM_OPTIONS: dict[str, str] = {
    "Nama Instansi": "Nama Instansi",
    "Nama Satuan Kerja": "Nama Satuan Kerja",
    "Sumber Dana": "Sumber Dana",
    "Tahun Anggaran": "Tahun Anggaran",
    "Cara Pengadaan": "Cara Pengadaan",
    "Jenis Pengadaan": "Jenis Pengadaan",
    "Metode Pengadaan": "Metode Pengadaan",
    "Produk Dalam Negeri": "Produk DN",
    "_nilai_bin": "Kelompok Nilai",
    "_nilai": "Total Nilai (kontinu)",
    "Kode RUP": "Kode RUP (temporal)",
}

# Dimensi default aktif untuk coloring
DIM_COLOR_DEFAULT = "_cluster_label"

# Kolom yang ditampilkan di tab Data
DATA_DISPLAY_COLS = [
    "Nama Paket",
    "Kode RUP",
    "Nama Instansi",
    "Nama Satuan Kerja",
    "Sumber Dana",
    "Tahun Anggaran",
    "Cara Pengadaan",
    "Jenis Pengadaan",
    "Metode Pengadaan",
    "Produk Dalam Negeri",
    "_nilai",
    "_nilai_bin",
    "_cluster_label",
]

# Kolom hover pada scatter plot
SCATTER_HOVER_COLS = [
    "Nama Paket",
    "Kode RUP",
    "Nama Instansi",
    "Nama Satuan Kerja",
    "Sumber Dana",
    "Tahun Anggaran",
    "Cara Pengadaan",
    "Jenis Pengadaan",
    "Metode Pengadaan",
    "Produk Dalam Negeri",
    "_nilai_bin",

]

# Kolom detail kluster di tab Statistik
CLUSTER_DETAIL_COLS = [
    "Nama Paket",
    "Kode RUP",
    "Nama Instansi",
    "Nama Satuan Kerja",
    "Sumber Dana",
    "Tahun Anggaran",
    "Cara Pengadaan",
    "Jenis Pengadaan",
    "Metode Pengadaan",
    "_nilai",
    "_nilai_bin",
]

# Kolom yang bisa dipilih di treemap
TREEMAP_GROUP_OPTIONS = [
    "_cluster_label",
    "Cara Pengadaan",
    "Jenis Pengadaan",
    "_nilai_bin",
]


# ══════════════════════════════════════════════════════════════
# UI — THEME & WARNA
# ══════════════════════════════════════════════════════════════

COLOR_BG = "#080810"
COLOR_SIDEBAR = "#10101c"
COLOR_BORDER = "#1c1c30"
COLOR_CARD_BG = "#0f0f1a"
COLOR_CARD_BG2 = "#141428"
COLOR_TEXT = "#b0b0cc"
COLOR_TEXT_DIM = "#44446a"
COLOR_ACCENT = "#f5a30a"  # oranye — highlight, aktif
COLOR_CYAN = "#00c8e8"  # biru cyan — metric value
COLOR_HEADER = "#eeeeff"

# Plotly chart palette
CHART_PALETTE = "Alphabet"  # px.colors.qualitative.Alphabet (26 warna)


# ══════════════════════════════════════════════════════════════
# STOPWORDS
# ══════════════════════════════════════════════════════════════

STOPWORDS_ID: set[str] = {
    # Generic Indonesian
    "dan",
    "atau",
    "yang",
    "dengan",
    "dalam",
    "dari",
    "pada",
    "ke",
    "di",
    "untuk",
    "oleh",
    "atas",
    "serta",
    "tidak",
    "secara",
    "lain",
    "ini",
    "itu",
    "juga",
    "akan",
    "telah",
    "dapat",
    "sudah",
    "ada",
    "hal",
    "agar",
    "yaitu",
    "adalah",
    "melalui",
    "sebagai",
    "sesuai",
    "antara",
    # Universal procurement prefixes — appear in >80% entries
    "belanja",
    "biaya",
    "pembayaran",
    "pengadaan",
    "penyediaan",
    "penyelenggaraan",
    "pelaksanaan",
    "pengelolaan",
    "penyusunan",
    "penyiapan",
    "penyampaian",
    "penyerahan",
    "penyempurnaan",
    "perencanaan",
    "pembinaan",
    "pendampingan",
    "pemeliharaan",
    "peningkatan",
    "pembangunan",
    "perbaikan",
    "persiapan",
    "pengembangan",
    "pembuatan",
    "pemutakhiran",
    # Broad org/location
    "kantor",
    "kegiatan",
    "unit",
    "dinas",
    "kerja",
    "daerah",
    "negeri",
    "umum",
    "tempat",
    "lainnya",
    "pusat",
    "nasional",
    "kementerian",
    "direktorat",
    "sekretariat",
    "jenderal",
    "kabupaten",
    "kota",
    "provinsi",
    "wilayah",
    "regional",
    "instansi",
    "satuan",
    "satker",
    "bidang",
    "bagian",
    # Generic object/material
    "alat",
    "bahan",
    "barang",
    "jasa",
    "modal",
    "peralatan",
    "perlengkapan",
    "sarana",
    "prasarana",
    "fasilitas",
    "infrastruktur",
    "material",
    # Meeting/event noise
    "rapat",
    "pertemuan",
    "koordinasi",
    "konsultasi",
    "sosialisasi",
    "bimtek",
    "workshop",
    "seminar",
    "pelatihan",
    "training",
    # Document noise
    "laporan",
    "dokumen",
    "administrasi",
    "naskah",
    "surat",
    "formulir",
    # Structural/fiscal noise
    "gol",
    "tarif",
    "tahun",
    "anggaran",
    "apbn",
    "apbd",
}
