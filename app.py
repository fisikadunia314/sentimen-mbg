"""Aplikasi web analisis sentimen komentar YouTube (Flask + Tailwind CSS).

Pipeline:
  Crawling (YouTube Data API v3)
    -> Preprocessing (Sastrawi: clean, normalisasi, stopword, stemming)
    -> Labeling leksikon (positif/negatif/netral)
    -> Training Naive Bayes & SVM
    -> Evaluasi (accuracy, macro F1, confusion matrix)
    -> Dashboard + analisis video + prediksi komentar baru
"""
import os
import json

import numpy as np
import pandas as pd
from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, jsonify, send_file,
)

from config import Config
from modules.crawler import crawl_from_url, crawl_mbg_data
from modules.preprocess import preprocess_dataframe
from modules.labeling import label_dataframe
from modules.modeling import train_models
from modules.evaluator import (
    predict_text, predict_batch,
    plot_sentiment_distribution, plot_confusion_matrix, plot_comment_vs_reply,
)
from modules.report import build_report_pdf

app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = Config.SECRET_KEY


# ==================== HELPER ====================
def _to_int(value) -> int:
    """Paksa nilai menjadi int (aman untuk string/None/NaN)."""
    if value is None:
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


@app.template_filter("num")
def num_fmt(value):
    """Filter tampilan: format angka dengan pemisah ribuan, aman utk string/NaN/None."""
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return "0"


def load_comments() -> pd.DataFrame:
    if os.path.exists(Config.COMMENTS_CSV):
        df = pd.read_csv(Config.COMMENTS_CSV)
        if "comment_text" in df.columns:
            df["comment_text"] = df["comment_text"].fillna("").astype(str)
        return df
    return pd.DataFrame()


def load_videos() -> pd.DataFrame:
    if os.path.exists(Config.VIDEOS_CSV):
        return pd.read_csv(Config.VIDEOS_CSV)
    return pd.DataFrame()


def load_metrics():
    if os.path.exists(Config.METRICS_JSON):
        with open(Config.METRICS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ===== Hasil analisis terakhir (agar tidak hilang saat refresh) =====
LAST_ANALYSIS_JSON = os.path.join(Config.DATA_DIR, "hasil_analisis.json")


def _py(v):
    """Konversi aman numpy/NaN -> tipe Python (untuk JSON)."""
    if isinstance(v, dict):
        return {k: _py(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_py(x) for x in v]
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if f != f else f  # NaN -> null
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def save_last_analysis(result: dict):
    try:
        with open(LAST_ANALYSIS_JSON, "w", encoding="utf-8") as f:
            json.dump(_py(result), f, ensure_ascii=False, default=str)
    except Exception as e:
        print(f"[WARN] simpan hasil analisis: {e}")


def load_last_analysis():
    if os.path.exists(LAST_ANALYSIS_JSON):
        try:
            with open(LAST_ANALYSIS_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] baca hasil analisis: {e}")
    return None


def chart_path(name: str) -> str:
    return os.path.join(Config.CHARTS_DIR, name)


def chart_exists(name: str) -> bool:
    return os.path.exists(chart_path(name))


def chart_ver(name: str) -> str:
    """Cache-busting sederhana berdasarkan mtime file chart."""
    p = chart_path(name)
    return str(int(os.path.getmtime(p))) if os.path.exists(p) else "0"


def plot_models(metrics: dict):
    """Generate confusion matrix charts dari metrik yang tersimpan."""
    for name in ["naive_bayes", "svm"]:
        if name in metrics and "confusion_matrix" in metrics[name]:
            try:
                plot_confusion_matrix(
                    metrics[name]["confusion_matrix"],
                    metrics[name]["labels"],
                    f"Confusion Matrix — {name.upper()}",
                    chart_path(f"cm_{name}.png"),
                )
            except Exception as e:  # chart gagal tidak boleh meruntuhkan route
                print(f"[WARN] plot confusion matrix {name}: {e}")


@app.context_processor
def inject_globals():
    return {
        "api_ready": bool(Config.YOUTUBE_API_KEY),
        "has_model": os.path.exists(Config.NB_MODEL),
    }


# ==================== ROUTES ====================
@app.route("/")
def index():
    df = load_comments()
    videos = load_videos()
    metrics = load_metrics()

    total = len(df)
    dist = {}
    if not df.empty and "label" in df.columns:
        dist = {k: int(v) for k, v in df["label"].value_counts().items()}
    top_sent = max(dist, key=dist.get) if dist else "-"

    return render_template(
        "index.html",
        total=total,
        dist=dist,
        top_sent=top_sent,
        metrics=metrics,
        has_data=not df.empty,
        has_videos=not videos.empty,
        video_rows=videos.head(10).to_dict(orient="records") if not videos.empty else [],
        dist_chart_ok=chart_exists("distribusi.png"),
        dist_chart_ver=chart_ver("distribusi.png"),
    )


# ==================== HELPER ANALISIS URL (dipakai /analyze-url & /laporan) ====================
def run_url_analysis(youtube_url: str, max_pages: int, include_replies: bool) -> dict:
    """Crawl + preprocess + label + (opsional) prediksi model untuk satu video.

    Mengembalikan dict `result`. Melempar ValueError untuk error user-friendly.
    """
    videos, comments = crawl_from_url(
        api_key=Config.YOUTUBE_API_KEY,
        url=youtube_url,
        max_pages=max_pages,
        include_replies=include_replies,
    )
    if comments.empty:
        raise ValueError("Video ditemukan, tetapi tidak ada komentar yang dapat dianalisis.")

    analyzed = preprocess_dataframe(comments, text_col="comment_text")
    if analyzed.empty:
        raise ValueError("Komentar ditemukan, tetapi tidak ada teks yang dapat dianalisis.")

    analyzed = label_dataframe(analyzed, tokens_col="norm_tokens")

    dist_all = {k: int(v) for k, v in analyzed["label"].value_counts().items()}
    comment_rows = analyzed[analyzed["type"] == "comment"] if "type" in analyzed.columns else analyzed
    reply_rows = analyzed[analyzed["type"] == "reply"] if "type" in analyzed.columns else analyzed.iloc[0:0]

    # Prediksi model (opsional — hanya jika model sudah dilatih)
    model_available = False
    agreement_nb = agreement_svm = None
    if os.path.exists(Config.NB_MODEL):
        try:
            pred_nb, pred_svm = predict_batch(
                analyzed["stemmed"].tolist(), Config.MODEL_DIR
            )
            analyzed["pred_nb"] = pred_nb
            analyzed["pred_svm"] = pred_svm
            agreement_nb = round(
                (analyzed["pred_nb"] == analyzed["label"]).mean() * 100, 1
            )
            agreement_svm = round(
                (analyzed["pred_svm"] == analyzed["label"]).mean() * 100, 1
            )
            model_available = True
        except Exception as e:
            print(f"[WARN] prediksi model: {e}")

    # Chart (di-generate sekali saat analisis, bukan per request)
    video_title = videos.iloc[0].get("title", "")
    short_title = (video_title[:60] + "…") if len(video_title) > 60 else video_title
    plot_sentiment_distribution(
        analyzed, "label", chart_path("distribusi_url.png"),
        title=f"Distribusi Sentimen — {short_title}" if short_title else "Distribusi Sentimen",
    )
    plot_comment_vs_reply(
        analyzed, Config.CHARTS_DIR,
        title=f"Komentar vs Reply — {short_title}" if short_title else None,
    )

    # Bersihkan kolom yang ditampilkan ke template (anti-NaN)
    sample_cols = ["comment_text", "type", "label", "score", "like_count"]
    if model_available:
        sample_cols += ["pred_nb", "pred_svm"]
    sample_df = analyzed[sample_cols].copy()
    sample_df["comment_text"] = sample_df["comment_text"].fillna("").astype(str)
    sample_df["like_count"] = (
        pd.to_numeric(sample_df["like_count"], errors="coerce").fillna(0).astype(int)
    )

    # Video info aman untuk template (NaN -> string kosong / angka 0)
    video_rec = {}
    for k, v in videos.iloc[0].to_dict().items():
        try:
            is_nan = bool(pd.isna(v))
        except (TypeError, ValueError):
            is_nan = False
        if k in ("view_count", "like_count", "comment_count"):
            video_rec[k] = _to_int(v)
        elif is_nan:
            video_rec[k] = ""
        else:
            video_rec[k] = v

    return {
        "url": youtube_url,
        "video": video_rec,
        "total": len(analyzed),
        "total_comment": len(comment_rows),
        "total_reply": len(reply_rows),
        "dist_all": dist_all,
        "dist_comment": {k: int(v) for k, v in comment_rows["label"].value_counts().items()},
        "dist_reply": {k: int(v) for k, v in reply_rows["label"].value_counts().items()},
        "model_available": model_available,
        "agreement_nb": agreement_nb,
        "agreement_svm": agreement_svm,
        "samples": sample_df.head(100).to_dict(orient="records"),
    }


@app.route("/analyze-url", methods=["GET", "POST"])
def analyze_url():
    # Tampilkan hasil analisis terakhir (agar tidak hilang saat refresh)
    result = load_last_analysis()
    if request.method == "POST":
        youtube_url = request.form.get("youtube_url", "").strip()
        try:
            if not Config.YOUTUBE_API_KEY:
                raise ValueError(
                    "YOUTUBE_API_KEY belum di-set. Salin .env.example menjadi .env lalu isi API key-nya."
                )
            if not youtube_url:
                raise ValueError("Masukkan URL video YouTube terlebih dahulu.")

            max_pages = int(request.form.get("max_pages", "5") or "5")
            if not 1 <= max_pages <= 20:
                raise ValueError("Maks. halaman komentar harus antara 1 dan 20.")

            result = run_url_analysis(
                youtube_url, max_pages,
                request.form.get("include_replies") == "on",
            )
            save_last_analysis(result)
            flash(f"Berhasil menganalisis {result['total']} komentar dan reply.", "success")
        except (TypeError, ValueError) as e:
            flash(str(e), "warning")
            result = load_last_analysis()
        except Exception as e:
            flash(f"Error saat analisis URL: {e}", "danger")
            result = load_last_analysis()

    return render_template(
        "analyze_url.html",
        result=result,
        dist_url_ok=chart_exists("distribusi_url.png"),
        dist_url_ver=chart_ver("distribusi_url.png"),
        vs_reply_ok=chart_exists("comment_vs_reply.png"),
        vs_reply_ver=chart_ver("comment_vs_reply.png"),
    )


@app.route("/crawl", methods=["GET", "POST"])
def crawl():
    if request.method == "POST":
        try:
            df_v, df_c = crawl_mbg_data(
                api_key=Config.YOUTUBE_API_KEY,
                queries=Config.SEARCH_QUERIES,
                max_videos=Config.MAX_VIDEOS_PER_QUERY,
                max_pages=Config.MAX_COMMENT_PAGES_PER_VIDEO,
                videos_csv=Config.VIDEOS_CSV,
                comments_csv=Config.COMMENTS_CSV,
            )
            flash(f"Berhasil crawl {len(df_c)} komentar dari {len(df_v)} video.", "success")
        except Exception as e:
            flash(f"Error saat crawling: {e}", "danger")
        return redirect(url_for("crawl"))

    df = load_comments()
    preview = df.head(20).to_dict(orient="records") if not df.empty else []
    return render_template("crawl.html", preview=preview, total=len(df))


@app.route("/preprocessing", methods=["GET", "POST"])
def preprocessing():
    df = load_comments()
    if df.empty or "comment_text" not in df.columns:
        flash("Belum ada data. Silakan crawl terlebih dahulu.", "warning")
        return redirect(url_for("crawl"))

    if request.method == "POST":
        try:
            df_proc = preprocess_dataframe(df, text_col="comment_text")
            df_proc = label_dataframe(df_proc, tokens_col="norm_tokens")
            df_proc.to_csv(Config.COMMENTS_CSV, index=False, encoding="utf-8-sig")

            # Regenerate chart distribusi (sekali, bukan per request)
            plot_sentiment_distribution(df_proc, "label", chart_path("distribusi.png"))
            flash(f"Preprocessing & labeling selesai: {len(df_proc)} komentar.", "success")
        except Exception as e:
            flash(f"Error preprocessing: {e}", "danger")
        return redirect(url_for("preprocessing"))

    has_proc = "stemmed" in df.columns
    preview = df.head(20).to_dict(orient="records")
    dist = {k: int(v) for k, v in df["label"].value_counts().items()} \
        if "label" in df.columns else {}
    return render_template(
        "preprocessing.html",
        preview=preview,
        total=len(df),
        has_proc=has_proc,
        dist=dist,
        dist_chart_ok=chart_exists("distribusi.png"),
        dist_chart_ver=chart_ver("distribusi.png"),
    )


@app.route("/training", methods=["GET", "POST"])
def training():
    df = load_comments()
    if df.empty or "label" not in df.columns:
        flash("Data belum diproses/dilabeli. Lakukan preprocessing dulu.", "warning")
        return redirect(url_for("preprocessing"))

    if request.method == "POST":
        try:
            train_models(
                df=df,
                text_col="stemmed",
                label_col="label",
                model_dir=Config.MODEL_DIR,
                metrics_path=Config.METRICS_JSON,
            )
            metrics = load_metrics()
            if metrics:
                plot_models(metrics)
            flash("Training selesai: Naive Bayes & SVM berhasil dilatih.", "success")
            return redirect(url_for("evaluation"))
        except Exception as e:
            flash(f"Error training: {e}", "danger")
        return redirect(url_for("training"))

    metrics = load_metrics()
    return render_template("training.html", metrics=metrics, total_rows=int(df.shape[0]))


@app.route("/evaluation")
def evaluation():
    metrics = load_metrics()
    if not metrics:
        flash("Belum ada hasil evaluasi. Latih model dulu.", "warning")
        return redirect(url_for("training"))

    plot_models(metrics)  # idempoten — hanya menulis file chart

    return render_template(
        "evaluation.html",
        metrics=metrics,
        cm_nb_ok=chart_exists("cm_naive_bayes.png"),
        cm_nb_ver=chart_ver("cm_naive_bayes.png"),
        cm_svm_ok=chart_exists("cm_svm.png"),
        cm_svm_ver=chart_ver("cm_svm.png"),
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    if request.method == "POST":
        text = request.form.get("komentar", "").strip()
        if not text:
            flash("Masukkan komentar terlebih dahulu.", "warning")
        elif len(text) > 2000:
            flash("Komentar terlalu panjang (maks. 2000 karakter).", "warning")
        else:
            try:
                result = predict_text(text, Config.MODEL_DIR)
            except FileNotFoundError:
                flash("Model belum dilatih. Silakan jalankan Training terlebih dahulu.", "warning")
            except Exception as e:
                flash(f"Error prediksi: {e}", "danger")
    return render_template("predict.html", result=result)


# ==================== LAPORAN PDF ====================
REPORT_PDF = os.path.join(Config.DATA_DIR, "laporan_sentimen.pdf")
REPORT_META = os.path.join(Config.DATA_DIR, "laporan_meta.json")
SETTINGS_JSON = os.path.join(Config.DATA_DIR, "pengaturan.json")

SETTING_FIELDS = ("nama", "nim", "prodi", "dosen")


def load_settings() -> dict:
    try:
        if os.path.exists(SETTINGS_JSON):
            with open(SETTINGS_JSON, encoding="utf-8") as f:
                d = json.load(f)
            return {k: str(d.get(k, "") or "") for k in SETTING_FIELDS}
    except Exception:
        pass
    return {k: "" for k in SETTING_FIELDS}


def save_settings(d: dict) -> None:
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    with open(SETTINGS_JSON, "w", encoding="utf-8") as f:
        json.dump({k: str(d.get(k, "") or "").strip() for k in SETTING_FIELDS},
                  f, ensure_ascii=False, indent=2)


@app.route("/laporan/settings", methods=["POST"])
def laporan_settings():
    current = load_settings()
    for k in SETTING_FIELDS:
        v = request.form.get(k, "").strip()
        if v:
            current[k] = v
    save_settings(current)
    flash("Identitas penulis tersimpan. Akan tampil di cover laporan.", "success")
    return redirect(url_for("laporan"))


@app.route("/laporan", methods=["GET", "POST"])
def laporan():
    last = load_last_analysis()

    if request.method == "POST":
        video_url = request.form.get("video_url", "").strip()
        try:
            # Jika diisi URL baru -> jalankan analisis dulu, lalu buat laporan
            if video_url:
                if not Config.YOUTUBE_API_KEY:
                    raise ValueError(
                        "YOUTUBE_API_KEY belum di-set. Salin .env.example menjadi .env lalu isi API key-nya."
                    )
                max_pages = int(request.form.get("max_pages", "5") or "5")
                if not 1 <= max_pages <= 20:
                    raise ValueError("Maks. halaman komentar harus antara 1 dan 20.")
                last = run_url_analysis(
                    video_url, max_pages,
                    request.form.get("include_replies") == "on",
                )
                save_last_analysis(last)

            if not last:
                raise ValueError(
                    "Belum ada hasil analisis. Jalankan Analisis Video terlebih dahulu "
                    "atau tempel URL video di sini."
                )

            # Bangun PDF
            out = build_report_pdf(
                REPORT_PDF,
                video=last.get("video", {}),
                analysis=last,
                metrics=load_metrics(),
                charts_dir=Config.CHARTS_DIR,
                settings=load_settings(),
            )
            vid = str(last.get("video", {}).get("video_id", "video"))
            with open(REPORT_META, "w", encoding="utf-8") as f:
                json.dump({"video_id": vid,
                           "name": f"laporan_sentimen_{vid}.pdf"}, f)
            flash("Laporan PDF berhasil dibuat dan diunduh.", "success")
            return send_file(
                out, mimetype="application/pdf", as_attachment=True,
                download_name=f"laporan_sentimen_{vid}.pdf",
            )
        except (TypeError, ValueError) as e:
            flash(str(e), "warning")
        except Exception as e:
            flash(f"Error membuat laporan: {e}", "danger")
        return redirect(url_for("laporan"))

    return render_template(
        "laporan.html",
        last=last,
        settings=load_settings(),
        metrics=load_metrics(),
        report_exists=os.path.exists(REPORT_PDF),
        report_name=(json.load(open(REPORT_META, encoding="utf-8")).get("name", "laporan_sentimen.pdf")
                     if os.path.exists(REPORT_META) else None),
    )


@app.route("/laporan/unduh")
def laporan_unduh():
    if not os.path.exists(REPORT_PDF):
        flash("Belum ada laporan. Buat laporan terlebih dahulu.", "warning")
        return redirect(url_for("laporan"))
    name = "laporan_sentimen.pdf"
    if os.path.exists(REPORT_META):
        try:
            name = json.load(open(REPORT_META, encoding="utf-8")).get("name", name)
        except Exception:
            pass
    return send_file(REPORT_PDF, mimetype="application/pdf",
                     as_attachment=True, download_name=name)


@app.route("/api/metrics")
def api_metrics():
    return jsonify(load_metrics() or {})


@app.route("/api/distribution")
def api_distribution():
    df = load_comments()
    if df.empty or "label" not in df.columns:
        return jsonify({})
    return jsonify({k: int(v) for k, v in df["label"].value_counts().items()})


@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(e):
    print(f"[ERROR] 500: {e}")
    return render_template("500.html"), 500


if __name__ == "__main__":
    app.run(debug=Config.DEBUG, host="0.0.0.0", port=5000)
