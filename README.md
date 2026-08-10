# Demo Sistem Rekomendasi Film Hybrid

Demo inferensi dari skripsi **"Implementasi Hybrid Filtering dan CatBoost
Reranker dalam Sistem Rekomendasi Film"** (Rizky Febrian Hidayat — 51422471,
Universitas Gunadarma).

Menggabungkan:
- **Content-Based Filtering** (TF-IDF + cosine similarity)
- **Collaborative Filtering** — Bayesian Personalized Ranking (BPR)
- **Meta-learner** — CatBoostRanker dengan fungsi kerugian YetiRank

pada dataset **MovieLens 20M**.

> Ini demo inferensi untuk keperluan akademik, bukan layanan produksi.
> Lihat Bab 3 skripsi (subbab Deployment) untuk cakupan penelitian.

---

## Cara menjalankan lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

Buka `http://localhost:8501` di browser.

Model (~CBF/BPR/YetiRank) diunduh otomatis dari HuggingFace Hub saat aplikasi
start: [`AwAkAXxX/hybrid-movie-recommender-catboost`](https://huggingface.co/AwAkAXxX/hybrid-movie-recommender-catboost).

---

## Deploy ke Streamlit Community Cloud

1. Buka [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. **New app** → pilih repo ini
4. Main file path: `app.py`
5. Klik **Deploy**

---

## Catatan teknis

Seluruh logika skor (`get_bpr_scores_batch`, `get_cbf_scores_batch`,
`recommend_top_n`) di `app.py` disalin **persis** dari fungsi
`score_unseen_catalog()` pada notebook `05_hybrid_yetirank_movielens20m.ipynb`,
untuk menjamin hasil demo konsisten dengan metodologi yang dilaporkan di
Bab 3 dan hasil evaluasi di Bab 4 skripsi.
