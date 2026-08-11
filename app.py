"""
Demo Sistem Rekomendasi Film Hybrid — CBF + BPR + CatBoostRanker (YetiRank)
Skripsi: Implementasi Hybrid Filtering dan CatBoost Reranker dalam Sistem
         Rekomendasi Film (MovieLens 20M)

Logika inferensi pada app ini mengikuti PERSIS fungsi score_unseen_catalog()
dari notebook 05_hybrid_yetirank_movielens20m.ipynb — bukan penulisan ulang
dari nol, untuk menjaga konsistensi antara skripsi dan demo.

Cara pakai:
1. Model diunduh runtime dari HuggingFace Hub (lihat HF_REPO_ID)
2. streamlit run app.py  (lokal) ATAU deploy ke Streamlit Community Cloud
"""

import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostRanker
from huggingface_hub import hf_hub_download
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# KONFIGURASI
# ============================================================
HF_REPO_ID = "AwAkAXxX/hybrid-movie-recommender-catboost"
HF_MODEL_FILES = (
    "models/preprocess_artifacts.pkl",
    "models/cbf_artifacts.pkl",
    "models/bpr_artifacts.pkl",
    "models/yetirank_artifacts.pkl",
    "models/yetirank_model.cbm",
)
TOP_N_DEFAULT = 10

st.set_page_config(page_title="Rekomendasi Film Hybrid", page_icon="🎬", layout="wide")


def restore_scaler(obj):
    """Pulihkan MinMaxScaler dari objek sklearn ATAU dict parameter (anti beda versi)."""
    if isinstance(obj, MinMaxScaler):
        return obj
    if isinstance(obj, dict) and obj.get("type") == "MinMaxScaler":
        sc = MinMaxScaler(clip=bool(obj.get("clip", False)))
        sc.data_min_ = np.asarray(obj["data_min_"])
        sc.data_max_ = np.asarray(obj["data_max_"])
        sc.scale_ = np.asarray(obj["scale_"])
        sc.min_ = np.asarray(obj["min_"])
        sc.n_features_in_ = int(obj["n_features_in_"])
        sc.n_samples_seen_ = obj.get("n_samples_seen_", 0)
        if obj.get("feature_names_in_") is not None:
            sc.feature_names_in_ = np.asarray(obj["feature_names_in_"])
        return sc
    raise TypeError(f"Format scaler tidak dikenali: {type(obj)}")


# ============================================================
# PEMUATAN ARTEFAK (disimpan di memori sesi — hanya sekali per server)
# ============================================================
@st.cache_resource(show_spinner="Mengunduh & memuat model hybrid dari HuggingFace Hub...")
def load_artifacts():
    paths = {}
    for repo_path in HF_MODEL_FILES:
        name = Path(repo_path).name
        paths[name] = hf_hub_download(repo_id=HF_REPO_ID, filename=repo_path)

    with open(paths["preprocess_artifacts.pkl"], "rb") as f:
        pp = pickle.load(f)
    with open(paths["cbf_artifacts.pkl"], "rb") as f:
        cbf = pickle.load(f)
    with open(paths["bpr_artifacts.pkl"], "rb") as f:
        bpr = pickle.load(f)
    with open(paths["yetirank_artifacts.pkl"], "rb") as f:
        yr = pickle.load(f)

    ranker = CatBoostRanker()
    ranker.load_model(paths["yetirank_model.cbm"])

    movies_catalog = pp["movies_catalog"].reset_index(drop=True)

    cosine_sim = cbf["cosine_sim"]
    mid2idx = cbf["mid2idx"]
    user_fav_idx = cbf["user_fav_idx"]

    # Artefak ringkas menyimpan faktor saja; artefak penuh punya objek bpr_model.
    if "bpr_model" in bpr:
        bpr_model = bpr["bpr_model"]
    else:
        bpr_model = SimpleNamespace(
            user_factors=np.asarray(bpr["user_factors"], dtype=np.float32),
            item_factors=np.asarray(bpr["item_factors"], dtype=np.float32),
        )
    user2idx = bpr["user2idx"]
    movie2idx = bpr["movie2idx"]
    user_train_mid = bpr["user_train_mid"]

    scaler = restore_scaler(yr["scaler"])
    features = yr["features"]
    num_features = yr["num_features"]

    # --- fitur film statis (identik dengan notebook 05, cell 5) ---
    if "n_ratings_train" not in movies_catalog.columns:
        ratings_train = pp.get("ratings_train")
        if ratings_train is None:
            st.error(
                "movies_catalog tidak punya statistik item dan ratings_train "
                "tidak tersedia di preprocess_artifacts.pkl (pakai versi ringkas)."
            )
            st.stop()
        stats = ratings_train.groupby("movieId").agg(
            n_ratings_train=("rating", "count"),
            avg_rating_train=("rating", "mean"),
            std_rating_train=("rating", "std"),
        ).fillna(0)
        movies_catalog = movies_catalog.merge(stats, on="movieId", how="left")

    movie_features = movies_catalog[
        ["movieId", "n_ratings_train", "avg_rating_train", "std_rating_train", "release_year", "genres"]
    ].copy()
    movie_features["std_rating_train"] = movie_features["std_rating_train"].fillna(0)
    movie_features["release_year"] = movie_features["release_year"].fillna(0).astype(int)
    movie_features["n_genres"] = movie_features["genres"].apply(
        lambda g: len(g.split("|")) if isinstance(g, str) and g != "(no genres listed)" else 0
    )
    movie_features = movie_features.set_index("movieId")
    all_movie_ids = movies_catalog["movieId"].astype(int).values

    return {
        "movies_catalog": movies_catalog,
        "movie_features": movie_features,
        "all_movie_ids": all_movie_ids,
        "cosine_sim": cosine_sim,
        "mid2idx": mid2idx,
        "user_fav_idx": user_fav_idx,
        "bpr_model": bpr_model,
        "user2idx": user2idx,
        "movie2idx": movie2idx,
        "user_train_mid": user_train_mid,
        "ranker": ranker,
        "scaler": scaler,
        "features": features,
        "num_features": num_features,
    }


# ============================================================
# FUNGSI SKOR — identik dengan notebook 03 / 04 / 05
# ============================================================
CBF_MAX_FAV = 100
CBF_TOP_K = 10


def get_bpr_scores_batch(art, uid, movie_ids):
    u = art["user2idx"].get(int(uid))
    if u is None:
        return np.zeros(len(movie_ids), dtype=np.float32)
    idxs = [art["movie2idx"].get(int(m)) for m in movie_ids]
    scores = np.zeros(len(movie_ids), dtype=np.float32)
    valid = [(j, i) for j, i in enumerate(idxs) if i is not None]
    if not valid:
        return scores
    item_idx = np.array([i for _, i in valid], dtype=np.int32)
    scores[[j for j, _ in valid]] = (
        art["bpr_model"].item_factors[item_idx] @ art["bpr_model"].user_factors[u]
    )
    return scores


def get_cbf_scores_batch(art, uid, catalog_indices):
    """Leave-one-out + rata-rata top-K similarity — identik notebook 03."""
    fav_all = np.array(art["user_fav_idx"].get(int(uid), []), dtype=np.int32)
    idx = np.asarray(catalog_indices, dtype=np.int32)
    if len(fav_all) == 0 or len(idx) == 0:
        return np.zeros(len(idx), dtype=np.float32)

    fav = fav_all[:CBF_MAX_FAV]
    sim = art["cosine_sim"][np.ix_(idx, fav)].astype(np.float32).copy()

    fav_pos = {int(f): j for j, f in enumerate(fav)}
    for row, cand_idx in enumerate(idx):
        col = fav_pos.get(int(cand_idx))
        if col is not None:
            sim[row, col] = -1.0

    k = min(CBF_TOP_K, sim.shape[1])
    if k <= 0:
        return np.zeros(len(idx), dtype=np.float32)
    topk = np.partition(sim, -k, axis=1)[:, -k:]
    return np.clip(topk.mean(axis=1), 0.0, 1.0).astype(np.float32)


def recommend_top_n(art, uid, top_n=10):
    """Mengikuti persis score_unseen_catalog() dari notebook 05."""
    seen = art["user_train_mid"].get(int(uid), set())
    pool = [m for m in art["all_movie_ids"] if m not in seen]
    if len(pool) < 2:
        return None

    bpr_s = get_bpr_scores_batch(art, uid, pool)
    cat_idx = np.array([art["mid2idx"].get(m, -1) for m in pool], dtype=np.int32)
    cbf_s = np.zeros(len(pool), dtype=np.float32)
    valid = cat_idx >= 0
    if valid.any():
        cbf_s[valid] = get_cbf_scores_batch(art, uid, cat_idx[valid])

    mf = art["movie_features"].reindex(pool).reset_index()
    mf["bpr_score"] = bpr_s
    mf["user_cbf_score"] = cbf_s

    feature_cols = list(art["features"])
    X = mf[feature_cols].fillna(0)
    Xs = X.copy()
    Xs[art["num_features"]] = art["scaler"].transform(X[art["num_features"]])
    hyb_score = art["ranker"].predict(Xs)

    result = mf[["movieId"] + feature_cols].copy()
    result["score_hybrid"] = hyb_score
    result["score_bpr"] = bpr_s
    result["score_cbf"] = cbf_s
    # Simpan juga fitur setelah MinMax (bahan langsung ke predict)
    for col in feature_cols:
        result[f"{col}__scaled"] = Xs[col].values
    result = result.merge(
        art["movies_catalog"][["movieId", "title", "genres"]], on="movieId", how="left"
    )
    return result.sort_values("score_hybrid", ascending=False).head(top_n).reset_index(drop=True)


# ============================================================
# UI
# ============================================================
def render_score_guide():
    """Dokumentasi umum arti skor pada hasil rekomendasi."""
    with st.expander("Dokumentasi skor hasil rekomendasi", expanded=False):
        st.markdown(
            """
Sistem menghitung skor untuk setiap film kandidat yang belum terdapat pada riwayat
rating pengguna terpilih, kemudian mengurutkan kandidat berdasarkan Skor Hybrid.

**1. Skor CBF (Content-Based Filtering)**  
Representasi film dibentuk dari konten (judul, genre, dan tag) melalui TF-IDF.
Kesamaan diukur dengan *cosine similarity* antara film kandidat dan himpunan film
favorit pengguna. Nilai akhir merupakan rata-rata **top-10** similarity, dengan
mekanisme leave-one-out apabila kandidat termasuk favorit. Skor dibatasi pada
rentang **0–1**. Semakin tinggi nilai, semakin tinggi kemiripan konten terhadap
profil favorit pengguna. Skor ini **bukan** persentase.

**2. Skor BPR (Bayesian Personalized Ranking)**  
Skor kolaboratif diperoleh dari faktor laten pengguna dan item. Nilai dihitung
sebagai **dot product** `item_factors[item] · user_factors[user]`. Skala tidak
dibatasi pada 0–1. Semakin tinggi nilai, semakin tinggi kecocokan menurut pola
preferensi kolektif. Skor ini **bukan** persentase.

**3. Skor Hybrid (CatBoostRanker — YetiRank)**  
Skor Hybrid merupakan keluaran meta-learner CatBoostRanker. Masukan model
meliputi skor BPR, skor CBF, serta fitur statistik film
(`n_ratings_train`, `avg_rating_train`, `std_rating_train`, `release_year`,
`n_genres`). Fitur numerik dinormalisasi dengan Min–Max scaler hasil pelatihan,
kemudian diprediksi oleh model YetiRank. Keluaran berupa **skor ranking** untuk
pengurutan kandidat, **bukan** probabilitas maupun persentase. Peringkat 1
ditentukan oleh Skor Hybrid tertinggi pada himpunan kandidat.

**Catatan:** urutan pada tabel hasil mengikuti Skor Hybrid secara menurun.
Skor CBF dan BPR disajikan sebagai komponen penjelas kontribusi konten dan
kolaboratif.
            """
        )


def render_result_explanation(uid, top_n, recs, feature_cols):
    """Ringkasan hasil Top-1 dan rincian fitur masukan Hybrid."""
    top = recs.iloc[0]
    title = top["title"]
    hyb = float(top["score_hybrid"])
    bpr = float(top["score_bpr"])
    cbf = float(top["score_cbf"])

    st.markdown("#### Ringkasan hasil peringkat 1")
    c1, c2, c3 = st.columns(3)
    c1.metric("Skor Hybrid", f"{hyb:.4f}")
    c2.metric("Skor BPR", f"{bpr:.4f}")
    c3.metric("Skor CBF", f"{cbf:.4f}")

    st.success(
        f"**Peringkat 1:** {title} (User ID {uid}). "
        f"Film ini menempati posisi teratas karena Skor Hybrid = {hyb:.4f}, "
        f"nilai tertinggi di antara {top_n} kandidat yang ditampilkan."
    )

    st.markdown(
        f"""
**Makna skor pada film peringkat 1**

| Skor | Nilai | Makna |
|---|---:|---|
| CBF | {cbf:.4f} | Rata-rata top-10 *cosine similarity* terhadap profil favorit pengguna (skala 0–1). Bukan persentase. |
| BPR | {bpr:.4f} | Dot product faktor laten pengguna–item pada model BPR. Bukan persentase. |
| Hybrid | {hyb:.4f} | Keluaran CatBoostRanker (YetiRank) untuk pengurutan kandidat. Bukan probabilitas/persentase. |

Urutan rekomendasi ditentukan oleh hubungan:  
Hybrid₁ ≥ Hybrid₂ ≥ … ≥ Hybrid_{top_n}.
        """
    )

    feature_labels = {
        "bpr_score": "Skor BPR (dot product faktor laten)",
        "user_cbf_score": "Skor CBF (rata-rata top-10 cosine similarity)",
        "n_ratings_train": "Jumlah rating film pada data latih",
        "avg_rating_train": "Rata-rata rating film pada data latih",
        "std_rating_train": "Simpangan baku rating film",
        "release_year": "Tahun rilis film",
        "n_genres": "Jumlah genre film",
    }

    rows = []
    for col in feature_cols:
        raw = float(top[col])
        scaled_col = f"{col}__scaled"
        scaled = float(top[scaled_col]) if scaled_col in top.index else float("nan")
        rows.append(
            {
                "Fitur": col,
                "Deskripsi": feature_labels.get(col, col),
                "Nilai mentah": raw,
                "Setelah MinMax": scaled,
            }
        )
    feat_df = pd.DataFrame(rows)

    st.markdown("#### Pembentukan Skor Hybrid")
    st.markdown(
        f"""
Skor Hybrid **tidak** dihitung sebagai penjumlahan langsung `BPR + CBF`.
Proses pembentukan skor untuk film *{title}* adalah sebagai berikut:

```text
tujuh fitur mentah → MinMaxScaler → CatBoostRanker.predict(...) → Skor Hybrid = {hyb:.4f}
```

Tabel berikut menampilkan vektor fitur yang menjadi masukan prediksi pada
film peringkat 1.
        """
    )
    st.dataframe(
        feat_df.style.format({"Nilai mentah": "{:.4f}", "Setelah MinMax": "{:.6f}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Kolom “Setelah MinMax” merupakan nilai yang diterima model YetiRank. "
        f"Skor Hybrid {hyb:.4f} merupakan keluaran ensemble pohon keputusan atas "
        "vektor tersebut, bukan hasil aritmetika manual antar fitur."
    )

    with st.expander("Spesifikasi fitur meta-learner"):
        st.markdown(
            """
Vektor fitur yang diproses CatBoostRanker (YetiRank) terdiri atas:

| Fitur | Sumber |
|---|---|
| `bpr_score` | Skor BPR (dot product faktor laten) |
| `user_cbf_score` | Skor CBF (rata-rata top-10 cosine similarity) |
| `n_ratings_train` | Jumlah rating film pada data latih |
| `avg_rating_train` | Rata-rata rating film pada data latih |
| `std_rating_train` | Simpangan baku rating film |
| `release_year` | Tahun rilis film |
| `n_genres` | Jumlah genre film |

Fitur numerik ditransformasi Min–Max menggunakan parameter scaler hasil
pelatihan, kemudian diprediksi oleh `CatBoostRanker.predict()` untuk
menghasilkan Skor Hybrid.
            """
        )


def main():
    st.title("🎬 Demo Sistem Rekomendasi Film Hybrid")
    st.caption(
        "Content-Based Filtering + Collaborative Filtering (BPR) → "
        "Meta-learner CatBoostRanker (YetiRank) — MovieLens 20M"
    )
    st.info(
        "Demo ini menampilkan inferensi dari model yang sudah dilatih pada skripsi. "
        "Bukan layanan produksi — lihat Bab 3 subbab Deployment untuk cakupan penelitian.",
        icon="ℹ️",
    )

    render_score_guide()

    art = load_artifacts()
    known_users = sorted(art["user2idx"].keys())
    # Dropdown penuh 138k user membuat Streamlit Cloud OOM/crash — tampilkan sampel.
    step = max(1, len(known_users) // 300)
    sample_users = known_users[::step][:300]

    col1, col2 = st.columns([2, 1])
    with col1:
        uid = st.selectbox(
            "Pilih User ID (sampel dari data latih)",
            options=sample_users,
            index=0,
            help=(
                f"Menampilkan {len(sample_users)} sampel dari "
                f"{len(known_users):,} pengguna data latih (batas RAM demo Cloud)."
            ),
        )
    with col2:
        top_n = st.slider("Jumlah rekomendasi (Top-N)", min_value=5, max_value=25, value=TOP_N_DEFAULT)

    if st.button("🔍 Buat Rekomendasi", type="primary"):
        with st.spinner("Menghitung skor CBF, BPR, dan menggabungkan lewat CatBoostRanker..."):
            recs = recommend_top_n(art, uid, top_n=top_n)

        if recs is None or recs.empty:
            st.warning("Tidak ada kandidat film unseen untuk pengguna ini.")
            return

        st.subheader(f"Top-{top_n} Rekomendasi untuk User {uid}")
        show = recs[["title", "genres", "score_hybrid", "score_bpr", "score_cbf"]].copy()
        show.columns = ["Judul Film", "Genre", "Skor Hybrid", "Skor BPR", "Skor CBF"]
        show.index = show.index + 1
        st.dataframe(
            show.style.format(
                {"Skor Hybrid": "{:.4f}", "Skor BPR": "{:.4f}", "Skor CBF": "{:.4f}"}
            ),
            use_container_width=True,
        )
        st.caption(
            "Urutan baris mengikuti Skor Hybrid secara menurun. "
            "Skor Hybrid, BPR, dan CBF bukan persentase."
        )

        render_result_explanation(uid, top_n, recs, feature_cols=art["features"])

        with st.expander("Lihat riwayat rating pengguna ini (data latih)"):
            seen = art["user_train_mid"].get(int(uid), set())
            hist = art["movies_catalog"][art["movies_catalog"]["movieId"].isin(seen)][
                ["title", "genres"]
            ].head(20)
            st.dataframe(hist, use_container_width=True)
            st.caption(f"Menampilkan 20 dari {len(seen)} film yang sudah dirating pengguna ini.")

    st.divider()
    st.caption(
        "Rizky Febrian Hidayat — 51422471 — Universitas Gunadarma. "
        "Model: CBF (TF-IDF + cosine similarity) · BPR (implicit) · "
        "CatBoostRanker (YetiRank) pada dataset MovieLens 20M."
    )


if __name__ == "__main__":
    main()
