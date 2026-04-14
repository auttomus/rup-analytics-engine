"""
clustering.py — TF-IDF, KMeans, HDBSCAN, c-TF-IDF labeling, analysis.
"""

import re
from functools import lru_cache
from collections import defaultdict
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.metrics import silhouette_score

from config.settings import (
    STOPWORDS_ID,
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_MAX_DF,
    TFIDF_MIN_DF_FLOOR, TFIDF_MIN_DF_RATIO, TFIDF_SUBLINEAR_TF,
    TFIDF_MIN_TOKEN_LEN,
    KMEANS_N_INIT, KMEANS_N_INIT_ANALYSIS,
    KMEANS_MAX_ITER, KMEANS_MAX_ITER_ANALYSIS, KMEANS_RANDOM_STATE,
    HDBSCAN_MIN_SAMPLES_RATIO, HDBSCAN_METRIC, HDBSCAN_CLUSTER_SELECTION,
    HDBSCAN_SWEEP_MIN, HDBSCAN_SWEEP_MAX_RATIO,
    HDBSCAN_SWEEP_FIXED, HDBSCAN_SWEEP_N_LOGSPACE, HDBSCAN_NOISE_THRESHOLD,
    SILHOUETTE_SAMPLE_SIZE, SILHOUETTE_HDBSCAN_SAMPLE_SIZE, SILHOUETTE_RANDOM_STATE,
    LABEL_EXEMPLAR_TOP_K, LABEL_MAX_WORDS, LABEL_MIN_CONFIDENCE, LABEL_GENERIC_TERMS,
    LABEL_TOP_N_TERMS, LABEL_MIN_TERM_SCORE, LABEL_EXAMPLE_MAX_WORDS, LABEL_NGRAM_RANGE
)


def _preprocess_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace('"', '').replace("'", "")
    text = re.sub(r'[/\-–—]', ' ', text)
    text = re.sub(r'[^a-z0-9\s_]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = text.split()
    if len(tokens) >= 4:
        deduped, i = [], 0
        while i < len(tokens):
            found = False
            for w in range(min(4, len(tokens) - i), 1, -1):
                if tokens[i:i+w] == tokens[max(0,i-w):i]:
                    i += w; found = True; break
            if not found:
                deduped.append(tokens[i]); i += 1
        tokens = deduped

    return ' '.join(t for t in tokens if len(t) > TFIDF_MIN_TOKEN_LEN)


def build_tfidf(texts: list[str]) -> tuple:
    processed = [_preprocess_text(t) for t in texts]
    n_docs    = len(processed)
    min_df    = max(TFIDF_MIN_DF_FLOOR, int(n_docs * TFIDF_MIN_DF_RATIO))

    vectorizer = TfidfVectorizer(
        stop_words   = list(STOPWORDS_ID),
        max_features = TFIDF_MAX_FEATURES,
        ngram_range  = TFIDF_NGRAM_RANGE,
        min_df       = min_df,
        max_df       = TFIDF_MAX_DF,
        sublinear_tf = TFIDF_SUBLINEAR_TF,
    )
    matrix = vectorizer.fit_transform(processed)
    return normalize(matrix, norm='l2', copy=True), vectorizer, processed


@lru_cache(maxsize=4)
def _get_sentence_model(model_name: str, device: str):
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name, device=device)
    except Exception as exc:
        raise RuntimeError(
            "Gagal memuat model embedding. Pastikan dependency `sentence-transformers` "
            "terpasang dan internet tersedia saat first-run download model."
        ) from exc


def build_text_embeddings(
    texts: list[str],
    model_name: str,
    device: str = "auto",
    batch_size: int = 64,
) -> tuple[sp.csr_matrix, None, list[str]]:
    processed = [_preprocess_text(t) for t in texts]
    use_device = device
    if device == "auto":
        try:
            import torch
            use_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            use_device = "cpu"

    model = _get_sentence_model(model_name, use_device)
    vectors = model.encode(
        processed,
        batch_size=max(1, int(batch_size)),
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    matrix = sp.csr_matrix(np.asarray(vectors, dtype=np.float32))
    return normalize(matrix, norm='l2', copy=True), None, processed


def build_tag_matrix(df: pd.DataFrame, dims: list[str]) -> sp.csr_matrix | None:
    if not dims:
        return None
    blocks = []
    for dim in dims:
        if dim not in df.columns:
            continue
        if dim == "_nilai":
            vals = np.log1p(df["_nilai"].fillna(0).values).astype(float)
            rng  = vals.max() - vals.min()
            if rng > 0:
                vals = (vals - vals.min()) / rng
            blocks.append(sp.csr_matrix(vals.reshape(-1, 1)))
        elif dim == "Kode RUP":
            vals = pd.to_numeric(df["Kode RUP"], errors="coerce").fillna(0).to_numpy(dtype=float)
            rng  = vals.max() - vals.min()
            if rng > 0:
                vals = (vals - vals.min()) / rng
            blocks.append(sp.csr_matrix(vals.reshape(-1, 1)))
        else:
            col     = df[dim].fillna("UNKNOWN").astype(str)
            le      = LabelEncoder()
            encoded = le.fit_transform(col)
            n_rows  = len(encoded)
            ohe     = sp.csr_matrix(
                (np.ones(n_rows), (np.arange(n_rows), encoded)),
                shape=(n_rows, len(le.classes_)),
            )
            blocks.append(normalize(ohe, norm='l2'))
    return sp.hstack(blocks, format='csr') if blocks else None


def combine_matrices(
    text_matrix: sp.csr_matrix | np.ndarray,
    tag_matrix: sp.csr_matrix | None,
    tfidf_weight: float = 0.5,
    tag_weight: float   = 0.5,
) -> sp.csr_matrix:
    text_csr = text_matrix if sp.issparse(text_matrix) else sp.csr_matrix(text_matrix)
    if tag_matrix is None:
        return text_csr
    combined = sp.hstack([
        text_csr.multiply(tfidf_weight),
        tag_matrix.multiply(tag_weight),
    ], format='csr')
    return normalize(combined, norm='l2', copy=True)


def run_kmeans(matrix, k: int) -> tuple:
    k     = max(2, min(k, matrix.shape[0]))
    model = KMeans(
        n_clusters   = k,
        n_init       = KMEANS_N_INIT,
        max_iter     = KMEANS_MAX_ITER,
        random_state = KMEANS_RANDOM_STATE,
    )
    return model.fit_predict(matrix), model


def run_hdbscan(matrix, min_cluster_size: int = 10) -> tuple:
    try:
        import hdbscan
        dense     = matrix.toarray() if sp.issparse(matrix) else matrix
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size     = min_cluster_size,
            min_samples          = max(1, min_cluster_size // HDBSCAN_MIN_SAMPLES_RATIO),
            metric               = HDBSCAN_METRIC,
            cluster_selection_method = HDBSCAN_CLUSTER_SELECTION,
        )
        labels = clusterer.fit_predict(dense)
        if -1 in labels:
            unique = [l for l in set(labels) if l != -1]
            if unique:
                centroids  = np.array([dense[labels == l].mean(axis=0) for l in unique])
                noise_mask = labels == -1
                dists      = np.linalg.norm(
                    dense[noise_mask][:, None] - centroids[None, :], axis=2
                )
                labels[noise_mask] = np.array(unique)[dists.argmin(axis=1)]
        return labels, None
    except ImportError:
        raise ImportError("hdbscan not installed. Run: pip install hdbscan")


def _ctfidf_labels(texts: list[str], labels: np.ndarray, n_top: int = 4) -> list[str]:
    cluster_ids  = sorted(int(c) for c in set(labels))
    k            = len(cluster_ids)
    cluster_docs = [
        ' '.join(texts[i] for i in range(len(texts)) if labels[i] == c)
        for c in cluster_ids
    ]
    count_vec    = CountVectorizer(stop_words=list(STOPWORDS_ID), ngram_range=(1,2), min_df=1)
    try:
        count_matrix = count_vec.fit_transform(cluster_docs)
    except ValueError:
        return ["LAINNYA" for _ in cluster_ids]
    vocab        = count_vec.get_feature_names_out()

    tf   = count_matrix.toarray().astype(float)
    tf  /= tf.sum(axis=1, keepdims=True).clip(min=1)
    idf  = np.log(1 + k / (count_matrix.toarray() > 0).sum(axis=0).clip(min=1))
    ctfidf   = tf * idf
    tag_mask = np.array([not v.startswith('tag_') for v in vocab])

    result = []
    for c in range(k):
        scores  = ctfidf[c] * tag_mask
        top_idx = scores.argsort()[-n_top:][::-1]
        terms   = [vocab[int(j)] for j in top_idx if scores[int(j)] > 0]
        result.append(" · ".join(terms[:3]).upper() if terms else "LAINNYA")
    return result


def _get_ngrams(text: str, n_range=(1, 2)) -> list[str]:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    tokens = [t for t in cleaned.split() if len(t) > 1]
    ngrams = []
    for n in range(n_range[0], n_range[1] + 1):
        for i in range(len(tokens) - n + 1):
            ngrams.append(" ".join(tokens[i:i+n]))
    return ngrams


def _build_exemplar_labels(
    texts: list[str],
    labels: np.ndarray,
    feature_matrix: sp.csr_matrix | np.ndarray,
) -> tuple[dict[int, str], dict[int, str], dict[int, float]]:
    mat = feature_matrix if sp.issparse(feature_matrix) else sp.csr_matrix(feature_matrix)
    cluster_ids = sorted(int(c) for c in set(labels))
    label_map: dict[int, str] = {}
    example_map: dict[int, str] = {}
    conf_map: dict[int, float] = {}

    for cluster_id in cluster_ids:
        idx = np.where(labels == cluster_id)[0]
        if len(idx) == 0:
            label_map[cluster_id] = "Lainnya"
            example_map[cluster_id] = "-"
            conf_map[cluster_id] = 0.0
            continue

        cluster_mat = mat[idx]
        centroid_arr = np.asarray(cluster_mat.mean(axis=0), dtype=np.float32).reshape(1, -1)
        centroid = normalize(centroid_arr, norm="l2", copy=True)
        sims = np.asarray(cluster_mat @ centroid.T).reshape(-1)
        
        top_k = max(1, min(LABEL_EXEMPLAR_TOP_K, len(idx)))
        ranked = np.argsort(-sims)[:top_k]

        term_scores = defaultdict(float)
        best_example = None
        best_example_score = -1.0

        for rank_pos in ranked:
            global_idx = int(idx[rank_pos])
            text = texts[global_idx]
            sim_score = float(sims[rank_pos])

            # Pilih exemplar terpendek yang memadai sebagai contoh
            word_count = len(text.split())
            if word_count <= LABEL_EXAMPLE_MAX_WORDS:
                # Preferensi pada kalimat ringkas namun dekat dengan centroid
                ex_score = sim_score - (word_count * 0.01)
                if ex_score > best_example_score:
                    best_example_score = ex_score
                    best_example = text

            ngrams = _get_ngrams(text, LABEL_NGRAM_RANGE)
            for term in ngrams:
                tokens = term.split()
                generic_count = sum(1 for t in tokens if t in LABEL_GENERIC_TERMS)
                generic_ratio = generic_count / max(1, len(tokens))
                length_penalty = 0.05 if len(tokens) < 2 else 0.0
                
                score = sim_score * (1.0 - 0.35 * generic_ratio) - length_penalty
                term_scores[term] += score

        # Ekstraksi final label dari top N kandidat tanpa overlap kata
        sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
        final_tokens = []
        total_conf = 0.0
        used_words = set()

        for term, score in sorted_terms:
            if score < LABEL_MIN_TERM_SCORE:
                continue
            term_words = set(term.split())
            if not term_words.intersection(used_words):
                final_tokens.append(term)
                used_words.update(term_words)
                total_conf += score
                if len(final_tokens) >= LABEL_TOP_N_TERMS:
                    break

        final_label = " · ".join(final_tokens).title() if final_tokens else "Lainnya"
        
        # Enforce hard length limit
        if len(final_label.split()) > LABEL_MAX_WORDS:
            final_label = " ".join(final_label.split()[:LABEL_MAX_WORDS])

        if not best_example:
            best_example = texts[int(idx[ranked[0]])]

        label_map[cluster_id] = final_label
        example_map[cluster_id] = best_example
        conf_map[cluster_id] = total_conf / max(1, len(final_tokens))

    return label_map, example_map, conf_map


def extract_cluster_labels(
    model, 
    vectorizer=None, 
    n_top: int = 3,
    texts: list[str] | None = None,
    labels: np.ndarray | None = None,
    feature_matrix: sp.csr_matrix | np.ndarray | None = None
) -> tuple[dict[int, str], dict[int, str]]:
    if texts is not None and labels is not None and feature_matrix is not None:
        cluster_ids = sorted(int(c) for c in set(labels))
        keyword_labels = _ctfidf_labels(texts, labels, n_top=n_top + 1)
        exemplar_map, example_map, conf_map = _build_exemplar_labels(texts, labels, feature_matrix)

        result_label_map: dict[int, str] = {}
        result_example_map: dict[int, str] = {}

        for i, cluster_id in enumerate(cluster_ids):
            keyword = keyword_labels[i].replace(" · ", " ").title() if i < len(keyword_labels) else "Lainnya"
            exemplar = exemplar_map.get(cluster_id, "Lainnya")
            conf = conf_map.get(cluster_id, 0.0)
            
            exemplar_tokens = set(exemplar.lower().split())
            keyword_tokens = set(keyword.lower().split())
            overlap = len(exemplar_tokens & keyword_tokens)
            
            # Kriteria gate kualitas
            if exemplar != "Lainnya" and (conf >= LABEL_MIN_CONFIDENCE or overlap >= 1):
                result_label_map[cluster_id] = exemplar
            else:
                result_label_map[cluster_id] = keyword
                
            result_example_map[cluster_id] = example_map.get(cluster_id, "-")
            
        return result_label_map, result_example_map
        
    # Fallback saat matrix/text absen
    if vectorizer is None:
        if hasattr(model, "n_clusters"):
            return {i: f"Kluster {i}" for i in range(int(model.n_clusters))}, {}
        return {0: "Lainnya"}, {}

    vocab    = vectorizer.get_feature_names_out()
    tag_mask = np.array([not v.startswith('tag_') for v in vocab])
    result_map: dict[int, str] = {}
    for i, center in enumerate(model.cluster_centers_):
        scores  = center * tag_mask
        top_idx = scores.argsort()[-n_top:][::-1]
        words   = [vocab[int(j)] for j in top_idx if scores[int(j)] > 0]
        result_map[i] = " · ".join(words).upper() if words else "LAINNYA"
    return result_map, {}


def compute_cluster_stats(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("_cluster_label")
        .agg(
            jumlah_paket=("_nilai", "count"),
            total_nilai =("_nilai", "sum"),
            rata_nilai  =("_nilai", "mean"),
            min_nilai   =("_nilai", "min"),
            max_nilai   =("_nilai", "max"),
        )
        .sort_values("total_nilai", ascending=False)
        .reset_index()
    )


def elbow_analysis(matrix, k_range: range) -> dict:
    n = matrix.shape[0]
    ks, inertias = [], []
    for k in k_range:
        if not (1 < k < n): continue
        km = KMeans(n_clusters=k, n_init=KMEANS_N_INIT_ANALYSIS,
                    max_iter=KMEANS_MAX_ITER_ANALYSIS, random_state=KMEANS_RANDOM_STATE,
                    algorithm="elkan")
        km.fit(matrix)
        ks.append(k); inertias.append(km.inertia_)

    best_k = ks[0]
    if len(inertias) >= 3:
        knee_idx = int(np.argmin(np.diff(np.diff(inertias)))) + 1
        best_k   = ks[knee_idx]
    return {"k": ks, "inertia": inertias, "best_k": best_k}


def silhouette_analysis(matrix, k_range: range) -> dict:
    n          = matrix.shape[0]
    rng        = np.random.default_rng(SILHOUETTE_RANDOM_STATE)
    sample_idx = rng.choice(n, size=min(n, SILHOUETTE_SAMPLE_SIZE), replace=False)
    mat_sample = (matrix[sample_idx].toarray() if sp.issparse(matrix)
                  else matrix[sample_idx])

    ks, scores = [], []
    for k in k_range:
        if not (1 < k < n): continue
        km  = KMeans(n_clusters=k, n_init=KMEANS_N_INIT_ANALYSIS,
                     max_iter=KMEANS_MAX_ITER_ANALYSIS, random_state=KMEANS_RANDOM_STATE,
                     algorithm="elkan")
        km.fit(matrix)
        lbl = km.labels_[sample_idx]
        ks.append(k)
        scores.append(
            float(silhouette_score(mat_sample, lbl, metric="euclidean"))
            if len(set(lbl)) >= 2 else 0.0
        )

    best_k = ks[int(np.argmax(scores))] if scores else (ks[0] if ks else 2)
    return {"k": ks, "score": scores, "best_k": best_k}


def hdbscan_min_size_analysis(matrix, min_sizes: list[int] | None = None) -> dict:
    try:
        import hdbscan as hdbscan_lib
    except ImportError:
        return {"error": "hdbscan not installed"}

    n = matrix.shape[0]
    if min_sizes is None:
        hi        = max(HDBSCAN_SWEEP_MIN + 1, int(n * HDBSCAN_SWEEP_MAX_RATIO))
        min_sizes = sorted(set(
            HDBSCAN_SWEEP_FIXED +
            list(np.unique(np.geomspace(HDBSCAN_SWEEP_MIN, hi,
                                        HDBSCAN_SWEEP_N_LOGSPACE).astype(int)))
        ))
        min_sizes = [s for s in min_sizes if s < n // 2]

    dense      = matrix.toarray() if sp.issparse(matrix) else matrix
    rng        = np.random.default_rng(SILHOUETTE_RANDOM_STATE)
    sample_idx = rng.choice(n, size=min(n, SILHOUETTE_HDBSCAN_SAMPLE_SIZE), replace=False)
    d_sample   = dense[sample_idx]
    sqrt_n     = int(np.sqrt(n))
    results    = []

    for mcs in min_sizes:
        try:
            lbl        = hdbscan_lib.HDBSCAN(
                min_cluster_size=mcs,
                min_samples=max(1, mcs // HDBSCAN_MIN_SAMPLES_RATIO),
                metric=HDBSCAN_METRIC,
                cluster_selection_method=HDBSCAN_CLUSTER_SELECTION,
            ).fit_predict(dense)
            noise_mask = lbl == -1
            unique     = [l for l in set(lbl) if l != -1]
            n_clusters = len(unique)
            noise_ratio= float(noise_mask.sum() / n)

            sil = 0.0
            if n_clusters >= 2:
                lbl_f = lbl.copy()
                if noise_mask.any():
                    c      = np.array([dense[lbl == cl].mean(axis=0) for cl in unique])
                    dists  = np.linalg.norm(dense[noise_mask][:,None] - c[None,:], axis=2)
                    lbl_f[noise_mask] = np.array(unique)[dists.argmin(axis=1)]
                ls = lbl_f[sample_idx]
                if len(set(ls)) >= 2:
                    sil = float(silhouette_score(d_sample, ls, metric="euclidean"))

            results.append({
                "min_size": mcs, "n_clusters": n_clusters,
                "noise_ratio": noise_ratio,
                "mean_cluster_size": (n - noise_mask.sum()) / max(n_clusters, 1),
                "silhouette": sil,
            })
        except Exception:
            continue

    viable = [r for r in results if r["noise_ratio"] < HDBSCAN_NOISE_THRESHOLD and r["n_clusters"] >= 2]
    best   = max(viable or results, key=lambda r: r["silhouette"], default={"min_size": sqrt_n})
    return {"results": results, "best_min_size": best["min_size"], "sqrt_n": sqrt_n}