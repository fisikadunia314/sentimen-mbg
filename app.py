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
import ast
import shutil
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd
from flask import (
    session,
    Flask, render_template, request, redirect,
    url_for, flash, jsonify, send_file,
)

from config import Config
from modules.crawler import (crawl_from_url, crawl_search_data,
                             CrawlNetworkError, friendly_network_message)
from modules.collector import collect_from_link, detect_source, CollectError
from modules.crawler_telegram import (
    crawl_telegram_data, TelegramCredsError, TELEGRAM_AVAILABLE,
)
from modules.crawler_instagram import crawl_instagram_data, InstagramCredsError
from modules.preprocess import (preprocess_dataframe, preprocess_text,
                                token_frequency, quality_report)
from modules import datasets as DS
from modules import dl_sequence as DL
from modules import transformer_model as TM
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
        # Kolom list (tokens/norm_tokens) tersimpan sebagai teks di CSV ->
        # kembalikan menjadi list sungguhan agar statistik & pipeline benar.
        for col in ("tokens", "norm_tokens"):
            if col in df.columns:
                df[col] = df[col].apply(_parse_token_cell)
        return df
    return pd.DataFrame()


def _parse_token_cell(value):
    """Ubah "[\'a\', \'b\']" dari CSV menjadi list; teks biasa -> split()."""
    if isinstance(value, (list, tuple)):
        return list(value)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    s = str(value).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(t) for t in parsed]
        except (ValueError, SyntaxError):
            pass
    return s.split()


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
LAST_CRAWL_JSON = os.path.join(Config.DATA_DIR, "queries_crawl.json")


def save_last_queries(queries: list):
    try:
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        with open(LAST_CRAWL_JSON, "w", encoding="utf-8") as f:
            json.dump(queries, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_last_queries():
    try:
        if os.path.exists(LAST_CRAWL_JSON):
            with open(LAST_CRAWL_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None


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


_LAST_ANALYSIS_DF = {"videos": None, "comments": None}


def save_dataset_safe(name: str, source: str, df_comments, df_videos=None,
                      note: str = ""):
    """Simpan dataset tanpa menggagalkan alur utama bila terjadi error."""
    try:
        if df_comments is None or len(df_comments) == 0:
            return None
        return DS.save_dataset(name, source, df_comments, df_videos, note=note)
    except Exception as e:
        print(f"[WARN] gagal simpan dataset: {e}")
        return None


def save_analysis_as_dataset(youtube_url: str, max_pages: int,
                             include_replies: bool, result: dict):
    """Simpan hasil run_url_analysis sebagai dataset baru (dengan label lengkap)."""
    analyzed = _LAST_ANALYSIS_DF.get("comments")
    videos = _LAST_ANALYSIS_DF.get("videos")
    if analyzed is None or analyzed.empty:
        return None
    title = (videos.iloc[0].get("title", "") if videos is not None
             and not videos.empty else "") or "Video YouTube"
    name = f"YouTube: {str(title)[:60]}"
    return DS.save_dataset(name, "youtube_url", analyzed, videos,
                           note=youtube_url)


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
    for name in ("naive_bayes", "svm", "lstm", "gru", "bert"):
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
        "telegram_pkg": TELEGRAM_AVAILABLE,
        "telegram_ready": bool(TELEGRAM_AVAILABLE and Config.TELEGRAM_API_ID
                               and Config.TELEGRAM_API_HASH
                               and os.path.exists(Config.TELEGRAM_SESSION)),
        "instagram_creds": bool(Config.INSTAGRAM_USERNAME and Config.INSTAGRAM_PASSWORD),
        "has_model": os.path.exists(Config.NB_MODEL),
        "torch_ready": DL.TORCH_AVAILABLE,
        "tf_ready": TM.TRANSFORMERS_AVAILABLE,
        "lang_active": current_lang(),
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

    # simpan untuk pembuatan dataset (tanpa crawl ulang -> hemat kuota API)
    _LAST_ANALYSIS_DF["videos"] = videos
    _LAST_ANALYSIS_DF["comments"] = analyzed

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
            # Simpan hasil analisis sebagai dataset tersendiri (tidak menimpa crawl)
            try:
                meta = save_analysis_as_dataset(youtube_url, max_pages,
                                                request.form.get("include_replies") == "on",
                                                result)
                if meta:
                    flash(f"Tersimpan sebagai dataset \"{meta['name']}\" "
                          f"({meta['comments']} komentar). Pilih di menu Preprocessing.",
                          "success")
            except Exception as e:
                print(f"[WARN] gagal simpan dataset analisis: {e}")
            flash(f"Berhasil menganalisis {result['total']} komentar dan reply.", "success")
        except CrawlNetworkError as e:
            flash(str(e), "danger")
            result = load_last_analysis()
        except (TypeError, ValueError) as e:
            flash(str(e), "warning")
            result = load_last_analysis()
        except Exception as e:
            msg = str(e)
            low = msg.lower()
            if any(k in low for k in ("10060", "10054", "timed out", "timeout",
                                      "connection", "getaddrinfo", "ssl")):
                flash(friendly_network_message(e), "danger")
            else:
                flash(f"Error saat analisis URL: {msg}", "danger")
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
        source = request.form.get("source", "youtube") or "youtube"
        if source not in ("youtube", "telegram", "instagram"):
            source = "youtube"

        # ---- parse keyword dari form (1 baris 1 kata kunci; juga terima pemisah koma/ titik koma) ----
        import re
        raw = request.form.get("queries", "")
        queries, seen = [], set()
        for part in re.split(r"[\n,;]+", raw):
            q = part.strip()
            if q and q.lower() not in seen:
                seen.add(q.lower())
                queries.append(q)
        queries = queries[:10]  # batas: 10 kata kunci

        if source == "youtube" and not queries:
            flash("Isi minimal satu kata kunci pencarian.", "warning")
            return redirect(url_for("crawl"))
        if source == "youtube":
            save_last_queries(queries)

        try:
            if source == "telegram":
                tg_channels = request.form.get("tg_channels", "") or ""
                tg_channel_list = [x.strip().lstrip("@") for x in re.split(r"[\n,;]+", tg_channels)]
                tg_channel_list = [x for x in tg_channel_list if x][:10]
                tg_max = int(request.form.get("tg_max_messages", "100") or "100")
                tg_replies = request.form.get("tg_include_replies") == "on"
                if not tg_channel_list:
                    flash("Isi minimal satu channel Telegram.", "warning")
                    return redirect(url_for("crawl"))
                df_v, df_c = crawl_telegram_data(
                    api_id=Config.TELEGRAM_API_ID,
                    api_hash=Config.TELEGRAM_API_HASH,
                    session_file=Config.TELEGRAM_SESSION,
                    channels=tg_channel_list,
                    max_messages=tg_max,
                    include_replies=tg_replies,
                    videos_csv=Config.VIDEOS_CSV,
                    comments_csv=Config.COMMENTS_CSV,
                )
                flash(f"Berhasil crawl (Telegram) {len(df_c)} pesan dari {len(df_v)} channel.", "success")
            elif source == "instagram":
                ig_mode = request.form.get("ig_mode", "hashtag") or "hashtag"
                ig_query = request.form.get("ig_query", "").strip().lstrip("#@")
                if not ig_query:
                    flash("Isi kata kunci hashtag atau nama profil Instagram.", "warning")
                    return redirect(url_for("crawl"))
                ig_max_posts = int(request.form.get("ig_max_posts", "5") or "5")
                ig_max_comments = int(request.form.get("ig_max_comments", "25") or "25")
                df_v, df_c = crawl_instagram_data(
                    username=Config.INSTAGRAM_USERNAME,
                    password=Config.INSTAGRAM_PASSWORD,
                    mode=ig_mode,
                    query=ig_query,
                    max_posts=ig_max_posts,
                    max_comments=ig_max_comments,
                    videos_csv=Config.VIDEOS_CSV,
                    comments_csv=Config.COMMENTS_CSV,
                )
                flash(f"Berhasil crawl (Instagram) {len(df_c)} komentar dari {len(df_v)} sumber.", "success")
            else:
                max_videos = int(request.form.get("max_videos", "5") or "5")
                max_pages = int(request.form.get("max_pages", "3") or "3")
                max_videos = max(1, min(10, max_videos))
                max_pages = max(1, min(20, max_pages))
                df_v, df_c = crawl_search_data(
                    api_key=Config.YOUTUBE_API_KEY,
                    queries=queries,
                    max_videos=max_videos,
                    max_pages=max_pages,
                    videos_csv=Config.VIDEOS_CSV,
                    comments_csv=Config.COMMENTS_CSV,
                )
                save_dataset_safe(f"YouTube crawl: {', '.join(queries)[:50]}",
                                  "youtube_crawl", df_c, df_v)
                flash(f"Berhasil crawl (YouTube) {len(df_c)} komentar dari {len(df_v)} video "
                      f"({len(queries)} kata kunci).", "success")
        except CrawlNetworkError as e:
            flash(f"Crawl berhenti karena jaringan: {e}", "danger")
        except Exception as e:
            msg = str(e)
            low = msg.lower()
            if any(k in low for k in ("10060", "10054", "timed out", "getaddrinfo")):
                flash(friendly_network_message(e), "danger")
            else:
                flash(f"Error saat crawling: {msg}", "danger")
        return redirect(url_for("crawl"))

    df = load_comments()
    preview = df.head(20).to_dict(orient="records") if not df.empty else []
    last_q = load_last_queries()

    # Sumber video (untuk validasi) — sanitasi NaN & format angka
    dfv = load_videos()
    videos_rows = []
    if not dfv.empty:
        for _, row in dfv.head(100).iterrows():
            def _s(v):
                return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
            def _n(v):
                try:
                    return f"{int(float(v)):,}"
                except (TypeError, ValueError):
                    return "0"
            url = _s(row.get("url"))
            if not url:
                vid = _s(row.get("video_id"))
                url = f"https://www.youtube.com/watch?v={vid}" if vid else ""
            videos_rows.append({
                "title": _s(row.get("title")),
                "channel": _s(row.get("channel")),
                "view_count": _n(row.get("view_count")),
                "comment_count": _n(row.get("comment_count")),
                "url": url,
            })

    return render_template(
        "crawl.html",
        preview=preview, total=len(df),
        videos_rows=videos_rows, videos_total=len(dfv),
        queries_value="\n".join(last_q or Config.SEARCH_QUERIES),
        last_queries=last_q,
    )


@app.route("/dataset", methods=["GET", "POST"])
def dataset_manager():
    """Manajemen dataset: pakai, ganti nama, hapus satu, atau reset semua."""
    if request.method == "POST":
        action = request.form.get("action", "")
        ds_id = request.form.get("dataset_id", "")

        if action == "delete" and ds_id:
            meta = DS.get_meta(ds_id)
            if DS.delete_dataset(ds_id):
                flash(f"Dataset \"{meta.get('name', ds_id)}\" dihapus.", "success")
            else:
                flash("Dataset tidak ditemukan.", "warning")

        elif action == "rename" and ds_id:
            DS.rename_dataset(ds_id, request.form.get("new_name", ""))
            flash("Nama dataset diperbarui.", "success")

        elif action == "reset_all":
            n = DS.delete_all()
            for path in (Config.COMMENTS_CSV, Config.VIDEOS_CSV, Config.METRICS_JSON,
                         LAST_ANALYSIS_JSON):
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass
            for folder in (Config.MODEL_DIR, Config.CHARTS_DIR):
                try:
                    for f in os.listdir(folder):
                        if f == ".gitkeep":
                            continue
                        p = os.path.join(folder, f)
                        if os.path.isfile(p):
                            os.remove(p)
                        else:
                            shutil.rmtree(p, ignore_errors=True)
                except Exception:
                    pass
            flash(f"Reset selesai: {n} dataset, data kerja, dan model dihapus. "
                  f"Silakan kumpulkan data baru dari nol.", "success")

        elif action == "use" and ds_id:
            dfc, dfv = DS.load_dataset(ds_id)
            if dfc is None:
                flash("Dataset tidak bisa dibaca.", "danger")
            else:
                os.makedirs(Config.DATA_DIR, exist_ok=True)
                dfc.to_csv(Config.COMMENTS_CSV, index=False, encoding="utf-8-sig")
                if dfv is not None and not dfv.empty:
                    dfv.to_csv(Config.VIDEOS_CSV, index=False, encoding="utf-8-sig")
                meta = DS.get_meta(ds_id)
                flash(f"Dataset \"{meta.get('name', ds_id)}\" dijadikan data aktif "
                      f"({len(dfc)} komentar).", "success")
                return redirect(url_for("preprocessing"))

        return redirect(url_for("dataset_manager"))

    active = load_comments()
    return render_template("dataset.html", datasets=DS.list_datasets(),
                           active_rows=int(len(active)) if not active.empty else 0,
                           has_model=os.path.exists(Config.NB_MODEL))


@app.route("/diagnostik", methods=["GET", "POST"])
def diagnostik():
    """Tes koneksi ke layanan yang dipakai aplikasi (bantu bedakan masalah jaringan)."""
    results = None
    if request.method == "POST":
        import socket as _socket
        import time as _time
        import urllib.request

        targets = [
            ("YouTube Data API", "youtube.googleapis.com"),
            ("Google umum", "www.google.com"),
            ("HuggingFace (model BERT)", "huggingface.co"),
            ("Instagram", "www.instagram.com"),
            ("Telegram API", "api.telegram.org"),
        ]
        results = []
        for label, host in targets:
            item = {"label": label, "host": host, "dns": False, "http": False,
                    "ms": None, "error": ""}
            # 1) DNS
            try:
                _socket.getaddrinfo(host, 443)
                item["dns"] = True
            except Exception as e:
                item["error"] = f"DNS gagal: {e}"
                results.append(item)
                continue
            # 2) HTTPS
            t0 = _time.time()
            try:
                req = urllib.request.Request(f"https://{host}/", method="HEAD",
                                             headers={"User-Agent": "sentimen-check/1.0"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    item["http"] = True
                    item["status"] = getattr(resp, "status", 200)
            except Exception as e:
                code = getattr(e, "code", None)
                if code:                       # server menjawab (mis. 403) = koneksi OK
                    item["http"] = True
                    item["status"] = code
                else:
                    item["error"] = str(e)[:160]
            item["ms"] = int((_time.time() - t0) * 1000)
            results.append(item)

        proxy = ", ".join(f"{k}={v}" for k, v in os.environ.items()
                          if k.lower() in ("http_proxy", "https_proxy", "no_proxy")) or ""
        flash("Tes koneksi selesai. Lihat tabel di bawah.", "success")
        return render_template("diagnostik.html", results=results, proxy=proxy,
                               timeout=os.getenv("HTTP_TIMEOUT", "60"))

    return render_template("diagnostik.html", results=None, proxy="",
                           timeout=os.getenv("HTTP_TIMEOUT", "60"))


@app.route("/koleksi", methods=["GET", "POST"])
def koleksi():
    """Data Koleksi: tempel SATU link -> komentar diambil -> tersimpan jadi CSV."""
    result = None
    form_url = ""
    if request.method == "POST":
        form_url = request.form.get("url", "").strip()
        try:
            max_comments = int(request.form.get("max_comments", "100") or "100")
            if not 1 <= max_comments <= 2000:
                raise ValueError("Maksimal komentar harus antara 1 dan 2000.")
            include_replies = request.form.get("include_replies") == "on"
            append_mode = request.form.get("append_mode") == "on"

            platform, dfv, dfc = collect_from_link(
                form_url,
                max_comments=max_comments,
                include_replies=include_replies,
                youtube_api_key=Config.YOUTUBE_API_KEY,
                ig_username=Config.INSTAGRAM_USERNAME,
                ig_password=Config.INSTAGRAM_PASSWORD,
                tg_api_id=Config.TELEGRAM_API_ID,
                tg_api_hash=Config.TELEGRAM_API_HASH,
                tg_session=Config.TELEGRAM_SESSION,
            )

            dfc["comment_id"] = dfc["comment_id"].astype(str)
            added = 0
            if append_mode:
                old_c = load_comments()
                old_v = load_videos()
                before = 0 if old_c.empty else len(old_c)
                if not old_c.empty:
                    old_c["comment_id"] = old_c["comment_id"].astype(str)
                    dfc = pd.concat([old_c, dfc], ignore_index=True, sort=False)
                if not old_v.empty:
                    dfv = pd.concat([old_v, dfv], ignore_index=True, sort=False)
                merged = dfc.drop_duplicates(subset=["comment_id"], keep="first")
                added = max(0, len(merged) - before)
                dfc = merged

            os.makedirs(Config.DATA_DIR, exist_ok=True)
            dfv.to_csv(Config.VIDEOS_CSV, index=False, encoding="utf-8-sig")
            dfc.to_csv(Config.COMMENTS_CSV, index=False, encoding="utf-8-sig")
            save_dataset_safe(f"Koleksi {platform}: {form_url[:45]}",
                              "koleksi", dfc, dfv, note=form_url)

            result = {
                "platform": platform,
                "title": str(dfv.iloc[0].get("title", "")) if len(dfv) else "",
                "url": form_url,
                "comments": int(len(dfc)),
                "added": int(added),
                "sources": int(len(dfv)),
                "preview": json.loads(dfc.head(15).to_json(orient="records")),
            }
            session["koleksi_result"] = result
            flash(
                f"Berhasil koleksi {len(dfc)} komentar dari {platform.upper()}"
                + (f" (data digabung, +{added} baru)" if append_mode else "")
                + ". Lanjut ke Preprocessing.",
                "success",
            )
        except (CollectError, ValueError) as e:
            flash(f"Gagal koleksi data: {e}", "danger")
        except Exception as e:
            flash(f"Error saat koleksi: {e}", "danger")
        return redirect(url_for("koleksi"))

    result = session.pop("koleksi_result", None)
    return render_template("koleksi.html", form_url=form_url, result=result,
                           yt_ready=bool(Config.YOUTUBE_API_KEY),
                           ig_ready=bool(Config.INSTAGRAM_USERNAME and Config.INSTAGRAM_PASSWORD),
                           tg_ready=bool(Config.TELEGRAM_API_ID and Config.TELEGRAM_API_HASH
                                         and os.path.exists(Config.TELEGRAM_SESSION)))


@app.route("/import", methods=["GET", "POST"])
def import_data():
    """Impor komentar manual (paste teks / upload CSV) -> CSV skema yang sama.

    Jalan keluar resmi saat crawling otomatis diblokir (mis. Instagram checkpoint).
    """
    TEXT_KEYS = ("comment_text", "komentar", "comment", "text", "teks", "isi", "isi_komentar")
    if request.method == "POST":
        label = request.form.get("source_label", "").strip() or "Impor Manual"
        src_url = request.form.get("source_url", "").strip()
        append_mode = request.form.get("append_mode") == "on"

        texts = []
        uploaded = request.files.get("csv_file")
        if uploaded and uploaded.filename:
            try:
                name = uploaded.filename.lower()
                if name.endswith((".csv", ".txt")):
                    df_in = pd.read_csv(uploaded, encoding="utf-8-sig")
                    col = next((c for c in df_in.columns
                                if str(c).strip().lower() in TEXT_KEYS), None)
                    if col is None and len(df_in.columns) == 1:
                        col = df_in.columns[0]
                    if col is None:
                        raise ValueError(
                            "Kolom teks komentar tidak ditemukan. "
                            "Beri nama kolom 'comment_text' atau 'komentar'.")
                    texts = df_in[col].dropna().astype(str).tolist()
                else:
                    raise ValueError("Hanya file .csv atau .txt yang didukung.")
            except Exception as e:
                flash(f"Gagal membaca file: {e}", "danger")
                return redirect(url_for("import_data"))

        if not texts:
            texts = request.form.get("pasted_text", "").splitlines()

        clean, seen = [], set()
        for t in texts:
            t = str(t).strip().strip('"').strip()
            if len(t) < 2:
                continue
            key = t.lower()
            if key in seen:
                continue
            seen.add(key)
            clean.append(t)

        if not clean:
            flash("Tidak ada komentar yang bisa diimpor. Tempel teks (1 komentar "
                  "per baris) atau unggah CSV.", "warning")
            return redirect(url_for("import_data"))

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        vid = "imp_" + (re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")[:24] or "manual")

        df_new = pd.DataFrame([{
            "comment_id": f"{vid}_{i + 1}",
            "parent_id": None,
            "type": "comment",
            "video_id": vid,
            "author": "manual",
            "comment_text": t,
            "published_at": now,
            "like_count": 0,
            "reply_count": 0,
        } for i, t in enumerate(clean)])

        df_vid = pd.DataFrame([{
            "video_id": vid,
            "title": label,
            "channel": "Impor manual",
            "published_at": now,
            "view_count": 0,
            "like_count": 0,
            "comment_count": len(df_new),
            "url": src_url,
        }])

        df_new["comment_id"] = df_new["comment_id"].astype(str)
        if append_mode:
            old_c = load_comments()
            if not old_c.empty:
                old_c["comment_id"] = old_c["comment_id"].astype(str)
                df_new = pd.concat([old_c, df_new], ignore_index=True, sort=False)
                df_new = df_new.drop_duplicates(subset=["comment_id"], keep="first")
            old_v = load_videos()
            if not old_v.empty:
                df_vid = pd.concat([old_v, df_vid], ignore_index=True, sort=False)

        os.makedirs(Config.DATA_DIR, exist_ok=True)
        df_new.to_csv(Config.COMMENTS_CSV, index=False, encoding="utf-8-sig")
        df_vid.to_csv(Config.VIDEOS_CSV, index=False, encoding="utf-8-sig")

        save_dataset_safe(f"Impor: {label}", "import", df_new, df_vid)
        flash(f"Berhasil impor {len(clean)} komentar dari '{label}'. "
              f"Selanjutnya buka Preprocessing.", "success")
        return redirect(url_for("crawl"))

    return render_template("import_data.html")


@app.route("/preprocessing", methods=["GET", "POST"])
def preprocessing():
    # Pilih dataset sumber (bisa sebelum ada data) ------------------------------
    if request.method == "POST" and request.form.get("step") == "pilih":
        chosen = request.form.getlist("dataset_ids")
        if request.form.get("use_all") == "on":
            chosen = [m["id"] for m in DS.list_datasets()]
        chosen = [d for d in chosen if d]
        if not chosen:
            flash("Pilih minimal satu dataset (atau centang \"gabung semua\").", "warning")
            return redirect(url_for("preprocessing"))
        df_src, df_vid = DS.load_combined(chosen)
        if df_src is None or df_src.empty:
            flash("Dataset yang dipilih tidak berisi komentar.", "warning")
            return redirect(url_for("preprocessing"))
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        df_src.to_csv(Config.COMMENTS_CSV, index=False, encoding="utf-8-sig")
        if df_vid is not None and not df_vid.empty:
            df_vid.to_csv(Config.VIDEOS_CSV, index=False, encoding="utf-8-sig")
        names = [DS.get_meta(d).get("name", d) for d in chosen]
        flash(f"Data aktif: {len(df_src)} komentar dari {len(chosen)} dataset "
              f"({'; '.join(n[:40] for n in names[:3])}"
              f"{'…' if len(names) > 3 else ''}).", "success")
        return redirect(url_for("preprocessing"))

    df = load_comments()

    if request.method == "POST":
        if df.empty or "comment_text" not in df.columns:
            flash("Belum ada data aktif. Pilih dataset terlebih dahulu.", "warning")
            return redirect(url_for("preprocessing"))
        try:
            lang = request.form.get("bahasa") or current_lang()
            if request.form.get("bahasa"):
                set_language(lang)
            # checkbox yang tidak dicentang TIDAK dikirim browser -> pakai penanda
            if request.form.get("sarcasm_form") == "1":
                handle_sarcasm = request.form.get("handle_sarcasm") == "1"
            else:
                handle_sarcasm = True
            df_proc = preprocess_dataframe(df, text_col="comment_text", lang=lang)
            df_proc = label_dataframe(df_proc, tokens_col="norm_tokens", lang=lang,
                                      handle_sarcasm=handle_sarcasm)
            df_proc.to_csv(Config.COMMENTS_CSV, index=False, encoding="utf-8-sig")

            plot_sentiment_distribution(df_proc, "label", chart_path("distribusi.png"))
            n_sarc = int(df_proc["sarkasme"].sum()) if "sarkasme" in df_proc.columns else 0
            extra = (f" {n_sarc} komentar terdeteksi SARKASME"
                     + (" (polaritas dibalik)" if handle_sarcasm else " (tidak dibalik)")
                     ) if n_sarc else ""
            flash(f"Preprocessing & labeling selesai: {len(df_proc)} komentar.{extra}",
                  "success")
        except Exception as e:
            flash(f"Error preprocessing: {e}", "danger")
        return redirect(url_for("preprocessing"))

    has_proc = "stemmed" in df.columns if not df.empty else False
    preview = df.head(20).to_dict(orient="records") if not df.empty else []
    top_tokens, quality = [], {}
    if has_proc:
        try:
            top_tokens = token_frequency(df, col="tokens", top_n=30)
            quality = quality_report(df)
            if "bahasa" in df.columns:
                quality["lang_dist"] = {str(k): int(v) for k, v in
                                        df["bahasa"].value_counts().items()}
        except Exception:
            top_tokens, quality = [], {}
    dist = {k: int(v) for k, v in df["label"].value_counts().items()} \
        if (not df.empty and "label" in df.columns) else {}
    return render_template(
        "preprocessing.html",
        preview=preview,
        total=len(df) if not df.empty else 0,
        has_proc=has_proc,
        dist=dist,
        dist_chart_ok=chart_exists("distribusi.png"),
        dist_chart_ver=chart_ver("distribusi.png"),
        top_tokens=top_tokens,
        quality=quality,
        lang=current_lang(),
        datasets=DS.list_datasets(),
    )


@app.route("/training", methods=["GET", "POST"])
def training():
    df = load_comments()
    if df.empty or "label" not in df.columns:
        flash("Data belum diproses/dilabeli. Lakukan preprocessing dulu.", "warning")
        return redirect(url_for("preprocessing"))

    if request.method == "POST":
        chosen = request.form.getlist("models") or ["klasik"]
        lang = request.form.get("bahasa") or current_lang()
        set_language(lang)
        done, errors = [], []

        # 1) Model klasik (Naive Bayes + SVM) — selalu tersedia
        if "klasik" in chosen:
            try:
                train_models(
                    df=df, text_col="stemmed", label_col="label",
                    model_dir=Config.MODEL_DIR, metrics_path=Config.METRICS_JSON,
                )
                done.append("Naive Bayes + SVM")
            except Exception as e:
                errors.append(f"Naive Bayes/SVM: {e}")

        # 2) LSTM / GRU
        for kind in ("lstm", "gru"):
            if kind not in chosen:
                continue
            try:
                m = DL.train_sequence_model(
                    df, text_col="stemmed", label_col="label",
                    model_dir=Config.MODEL_DIR, kind=kind)
                merge_metrics(kind, m)
                done.append(kind.upper())
            except Exception as e:
                errors.append(f"{kind.upper()}: {e}")

        # 3) BERT / IndoBERT (termasuk dataset campuran)
        if "bert" in chosen:
            try:
                if lang == "mix":
                    m = TM.train_bert_mixed(
                        df, text_col="comment_text", label_col="label",
                        base_dir=Config.MODEL_DIR)
                    merge_metrics("bert", m,
                                  extra={"base_model": m.get("base_model", ""),
                                         "mixed": True})
                    done.append("BERT campuran (ID+EN)")
                else:
                    m = TM.train_bert(
                        df, text_col="comment_text", label_col="label",
                        base_dir=Config.MODEL_DIR, lang=lang)
                    merge_metrics("bert", m,
                                  extra={"base_model": m.get("base_model", "")})
                    done.append("BERT/IndoBERT" if lang == "id" else "BERT")
            except Exception as e:
                errors.append(f"BERT: {e}")

        if done:
            flash("Training selesai: " + ", ".join(done) + ".", "success")
            metrics = load_metrics()
            if metrics:
                plot_models(metrics)
            if errors:
                flash("Sebagian model gagal — " + " | ".join(errors), "warning")
            return redirect(url_for("evaluation"))
        flash("Training gagal: " + (" | ".join(errors) or "tidak ada model dipilih"), "danger")
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
            lang = current_lang()
            try:
                result = predict_text(text, Config.MODEL_DIR)
            except FileNotFoundError:
                result = None
            except Exception as e:
                result = None
                flash(f"Error prediksi model klasik: {e}", "danger")

            # Model tambahan (opsional — hanya kalau sudah dilatih)
            extra_pred = {}
            stemmed = (result or {}).get("stemmed") or \
                preprocess_text(text, lang)["stemmed"]

            # deteksi sarkasme untuk teks yang diuji
            sarc_info = {"sarkasme": False, "skor": 0, "petunjuk": []}
            try:
                from modules.sarcasm import analyze as _sarc_analyze
                sarc_info = _sarc_analyze(text)
            except Exception:
                pass
            for kind, label in (("lstm", "LSTM"), ("gru", "GRU")):
                if DL.TORCH_AVAILABLE and DL.is_trained(Config.MODEL_DIR, kind):
                    try:
                        lab, conf = DL.predict_one(stemmed, Config.MODEL_DIR, kind)
                        extra_pred[label] = {"label": lab, "confidence": round(conf, 3)}
                    except Exception:
                        pass
            detected = None
            if TM.TRANSFORMERS_AVAILABLE:
                try:
                    if lang == "mix" and TM.is_trained_mix(Config.MODEL_DIR):
                        from modules.lang_detect import detect_language
                        detected = detect_language(text)
                        lab, conf, used = TM.predict_one_mixed(text, Config.MODEL_DIR)
                        extra_pred["BERT campuran"] = {
                            "label": lab, "confidence": round(conf, 3),
                            "lang": used}
                    elif TM.is_trained(Config.MODEL_DIR, lang):
                        lab, conf = TM.predict_one(text, Config.MODEL_DIR, lang)
                        extra_pred["BERT/IndoBERT" if lang == "id" else "BERT"] = {
                            "label": lab, "confidence": round(conf, 3)}
                except Exception:
                    pass

            if result is None and not extra_pred:
                flash("Model belum dilatih. Silakan jalankan Training terlebih dahulu.",
                      "warning")
            elif result is not None:
                result["extra"] = extra_pred
                result["sarkasme"] = sarc_info
            else:
                result = {"text": text, "extra": extra_pred,
                          "cleaned": preprocess_text(text, lang)["clean"],
                          "stemmed": stemmed}
    return render_template("predict.html", result=result,
                           dl_ready=DL.TORCH_AVAILABLE,
                           bert_ready=TM.TRANSFORMERS_AVAILABLE,
                           lang_now=current_lang())


# ==================== LAPORAN PDF ====================
REPORT_PDF = os.path.join(Config.DATA_DIR, "laporan_sentimen.pdf")
REPORT_META = os.path.join(Config.DATA_DIR, "laporan_meta.json")
SETTINGS_JSON = os.path.join(Config.DATA_DIR, "pengaturan.json")

SETTING_FIELDS = ("nama", "nim", "prodi", "dosen", "bahasa")


def current_lang() -> str:
    """Bahasa aktif: 'id', 'en', atau 'mix' — disimpan di data/pengaturan.json."""
    lang = str(load_settings().get("bahasa", "id") or "id").lower()
    if lang.startswith("mix"):
        return "mix"
    return "en" if lang.startswith("en") else "id"


def load_settings() -> dict:
    try:
        if os.path.exists(SETTINGS_JSON):
            with open(SETTINGS_JSON, encoding="utf-8") as f:
                d = json.load(f)
            return {k: str(d.get(k, "") or "") for k in SETTING_FIELDS}
    except Exception:
        pass
    return {k: "" for k in SETTING_FIELDS}


def set_language(lang: str) -> None:
    """Simpan bahasa aktif ke pengaturan."""
    cur = load_settings()
    low = str(lang).lower()
    cur["bahasa"] = ("mix" if low.startswith("mix")
                     else "en" if low.startswith("en") else "id")
    save_settings(cur)


def merge_metrics(name: str, new_metrics: dict, extra: dict = None) -> None:
    """Gabungkan metrik model DL ke metrics.json (tanpa menimpa NB/SVM)."""
    metrics = load_metrics() or {}
    metrics[name] = new_metrics
    if extra:
        metrics[name].update(extra)
    with open(Config.METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


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
    # Lokal: python app.py (port 5000). Deploy: gunakan wsgi.py + gunicorn.
    app.run(debug=Config.DEBUG, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
