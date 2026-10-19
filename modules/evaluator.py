"""Modul prediksi komentar baru & visualisasi (v2).

Perbaikan:
- Model di-cache (bukan load ulang 3 pickle per request).
- Judul chart distribusi bisa diparameterisasi (untuk analisis URL tertentu).
- predict_text memberikan pesan ramah jika model belum dilatih.
"""
import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .preprocess import clean_text, normalize_text, remove_stopwords, stem_text
from .lexicon import score_text, label_from_score

# Sederhana: cache model per model_dir (key = mtime file model)
_MODEL_CACHE = {}


def _load_models(model_dir: str):
    """Load model + vectorizer (di-cache sampai file model berubah)."""
    nb_path = os.path.join(model_dir, "naive_bayes.pkl")
    svm_path = os.path.join(model_dir, "svm.pkl")
    tfidf_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")

    for p in (nb_path, svm_path, tfidf_path):
        if not os.path.exists(p):
            raise FileNotFoundError("Model belum dilatih. Silakan jalankan Training terlebih dahulu.")

    key = (model_dir,
           os.path.getmtime(nb_path),
           os.path.getmtime(svm_path),
           os.path.getmtime(tfidf_path))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = (
            joblib.load(nb_path),
            joblib.load(svm_path),
            joblib.load(tfidf_path),
        )
        # Bersihkan entri lama agar memori tidak membengkak
        old = [k for k in _MODEL_CACHE if k[0] == model_dir and k != key]
        for k in old:
            _MODEL_CACHE.pop(k, None)
    return _MODEL_CACHE[key]


def predict_text(text: str, model_dir: str) -> dict:
    """Prediksi satu komentar dengan NB & SVM + leksikon."""
    nb, svm, tfidf = _load_models(model_dir)

    # Preprocessing (urutan sama persis dengan pipeline training)
    cleaned = clean_text(text)
    normalized = normalize_text(cleaned)
    no_stop = remove_stopwords(normalized)
    stemmed = stem_text(no_stop)
    tokens = stemmed.split()
    norm_tokens = no_stop.split()

    vec = tfidf.transform([stemmed])
    pred_nb = str(nb.predict(vec)[0])
    pred_svm = str(svm.predict(vec)[0])

    # Lexicon-based (pada token normalisasi, belum di-stem)
    lex_score = score_text(norm_tokens)
    lex_label = label_from_score(lex_score)

    return {
        "text": text,
        "cleaned": cleaned,
        "stemmed": stemmed,
        "naive_bayes": pred_nb,
        "svm": pred_svm,
        "lexicon": lex_label,
        "lexicon_score": lex_score,
    }


def predict_batch(texts: list, model_dir: str):
    """Prediksi banyak teks sekaligus (untuk hasil analisis URL)."""
    nb, svm, tfidf = _load_models(model_dir)
    vec = tfidf.transform([t.astype(str) if hasattr(t, "astype") else str(t) for t in texts])
    return [str(p) for p in nb.predict(vec)], [str(p) for p in svm.predict(vec)]


# ==================== VISUALISASI ====================
def plot_sentiment_distribution(df, label_col: str, out_path: str, title: str = None):
    title = title or "Distribusi Sentimen Komentar YouTube"
    plt.figure(figsize=(7, 4))
    order = ["positif", "netral", "negatif"]
    counts = df[label_col].value_counts().reindex(order).fillna(0).astype(int)
    colors = ["#10b981", "#94a3b8", "#ef4444"]
    sns.barplot(x=counts.index, y=counts.values, palette=colors, hue=counts.index,
                legend=False)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel("Kelas Sentimen")
    plt.ylabel("Jumlah Komentar")
    for i, v in enumerate(counts.values):
        plt.text(i, v + max(counts.values) * 0.02, str(int(v)),
                 ha="center", fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_confusion_matrix(cm, labels, title: str, out_path: str):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(title, fontsize=11, fontweight="bold")
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_comment_vs_reply(df, charts_dir: str, title: str = None):
    """Grafik perbandingan sentimen komentar top-level vs reply."""
    if "type" not in df.columns or "label" not in df.columns:
        return
    title = title or "Sentimen Komentar Top-Level vs Reply"
    plt.figure(figsize=(8, 4))
    order = ["positif", "netral", "negatif"]
    pivot = df.groupby(["type", "label"]).size().unstack(fill_value=0).reindex(
        columns=order, fill_value=0
    )
    pivot.plot(kind="bar", color=["#10b981", "#94a3b8", "#ef4444"], ax=plt.gca())
    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel("Tipe")
    plt.ylabel("Jumlah")
    plt.xticks(rotation=0)
    plt.legend(title="Sentimen")
    plt.tight_layout()
    out = os.path.join(charts_dir, "comment_vs_reply.png")
    os.makedirs(charts_dir, exist_ok=True)
    plt.savefig(out, dpi=120)
    plt.close()
