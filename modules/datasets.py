"""Manajemen dataset: setiap koleksi disimpan TERPISAH dengan nama & sumber.

Struktur penyimpanan:
    data/datasets/<id>/
        meta.json       -> id, nama, sumber, dibuat, jumlah, bahasa
        comments.csv    -> skema kolom komentar (sama seperti crawler)
        videos.csv      -> skema kolom sumber/video

Manfaat:
  * Hasil Analisis Video TIDAK lagi menimpa dataset hasil Crawling.
  * Di halaman Preprocessing, pengguna memilih dataset mana yang dipakai —
    satu dataset, beberapa dataset, atau gabung semua.
  * Dataset bisa dihapus satu-satu, atau semua (mulai dari nol).

Modul ini tidak bergantung pada app.py agar mudah diuji.
"""
import json
import os
import re
import shutil
import uuid
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "datasets")

COMMENT_COLS = ["comment_id", "parent_id", "type", "video_id", "author",
                "comment_text", "published_at", "like_count", "reply_count"]
VIDEO_COLS = ["video_id", "title", "channel", "published_at",
              "view_count", "like_count", "comment_count", "url"]

SOURCE_LABELS = {
    "youtube_url": "YouTube — Analisis Video (1 link)",
    "youtube_crawl": "YouTube — Crawling kata kunci",
    "telegram": "Telegram — Crawling channel",
    "instagram": "Instagram — Crawling",
    "koleksi": "Data Koleksi (link)",
    "import": "Impor Manual",
    "gabungan": "Gabungan beberapa dataset",
}


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return s[:40] or "dataset"


def ensure_base():
    os.makedirs(BASE_DIR, exist_ok=True)


def _dir(ds_id: str) -> str:
    return os.path.join(BASE_DIR, ds_id)


def new_id(name: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{_slug(name)}-{uuid.uuid4().hex[:4]}"


def save_dataset(name: str, source: str, df_comments: pd.DataFrame,
                 df_videos: pd.DataFrame = None, note: str = "") -> dict:
    """Simpan koleksi sebagai dataset baru. Return metadata-nya."""
    ensure_base()
    if df_comments is None:
        df_comments = pd.DataFrame(columns=COMMENT_COLS)
    df_comments = df_comments.copy()
    for col in COMMENT_COLS:
        if col not in df_comments.columns:
            df_comments[col] = None
    df_comments = df_comments[COMMENT_COLS]
    df_comments["comment_id"] = df_comments["comment_id"].astype(str)

    if df_videos is None:
        df_videos = pd.DataFrame(columns=VIDEO_COLS)
    df_videos = df_videos.copy()
    for col in VIDEO_COLS:
        if col not in df_videos.columns:
            df_videos[col] = None
    df_videos = df_videos[VIDEO_COLS]

    ds_id = new_id(name)
    d = _dir(ds_id)
    os.makedirs(d, exist_ok=True)
    df_comments.to_csv(os.path.join(d, "comments.csv"), index=False,
                       encoding="utf-8-sig")
    df_videos.to_csv(os.path.join(d, "videos.csv"), index=False,
                     encoding="utf-8-sig")

    meta = {
        "id": ds_id,
        "name": str(name).strip() or "Tanpa nama",
        "source": source,
        "source_label": SOURCE_LABELS.get(source, source),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "comments": int(len(df_comments)),
        "sources_count": int(len(df_videos)),
        "note": note[:300],
    }
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def list_datasets() -> list:
    """Semua dataset, terbaru di atas."""
    ensure_base()
    out = []
    for name in os.listdir(BASE_DIR):
        d = os.path.join(BASE_DIR, name)
        meta_path = os.path.join(d, "meta.json")
        if not os.path.isdir(d) or not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            if not os.path.exists(os.path.join(d, "comments.csv")):
                continue
            out.append(meta)
        except Exception:
            continue
    out.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return out


def get_meta(ds_id: str) -> dict:
    path = os.path.join(_dir(ds_id), "meta.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_dataset(ds_id: str):
    """Return (df_comments, df_videos) atau (None, None) bila tidak ada."""
    d = _dir(ds_id)
    cpath = os.path.join(d, "comments.csv")
    if not os.path.exists(cpath):
        return None, None
    try:
        dfc = pd.read_csv(cpath)
        vpath = os.path.join(d, "videos.csv")
        dfv = pd.read_csv(vpath) if os.path.exists(vpath) else pd.DataFrame()
        return dfc, dfv
    except Exception:
        return None, None


def load_combined(ds_ids: list):
    """Gabungkan beberapa dataset (dedup komentar per video_id+comment_id)."""
    frames_c, frames_v = [], []
    for ds_id in ds_ids:
        dfc, dfv = load_dataset(ds_id)
        if dfc is not None and not dfc.empty:
            frames_c.append(dfc)
        if dfv is not None and not dfv.empty:
            frames_v.append(dfv)
    if not frames_c:
        return pd.DataFrame(columns=COMMENT_COLS), pd.DataFrame(columns=VIDEO_COLS)

    dfc = pd.concat(frames_c, ignore_index=True, sort=False)
    dfc["comment_id"] = dfc["comment_id"].astype(str)
    # dedup: id komentar unik per video (id Telegram/Instagram bisa kembar antar sumber)
    if "video_id" in dfc.columns:
        dfc = dfc.drop_duplicates(subset=["video_id", "comment_id"], keep="first")
    else:
        dfc = dfc.drop_duplicates(subset=["comment_id"], keep="first")

    dfv = (pd.concat(frames_v, ignore_index=True, sort=False)
           if frames_v else pd.DataFrame(columns=VIDEO_COLS))
    if not dfv.empty and "video_id" in dfv.columns:
        dfv = dfv.drop_duplicates(subset=["video_id"], keep="first")
    return dfc.reset_index(drop=True), dfv.reset_index(drop=True)


def delete_dataset(ds_id: str) -> bool:
    d = _dir(ds_id)
    if os.path.isdir(d) and os.path.abspath(d).startswith(os.path.abspath(BASE_DIR)):
        shutil.rmtree(d, ignore_errors=True)
        return True
    return False


def delete_all() -> int:
    """Hapus SEMUA dataset (mulai dari nol). Return jumlah yang dihapus."""
    ensure_base()
    n = 0
    for name in os.listdir(BASE_DIR):
        d = os.path.join(BASE_DIR, name)
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
            n += 1
    return n


def rename_dataset(ds_id: str, new_name: str) -> bool:
    path = os.path.join(_dir(ds_id), "meta.json")
    if not os.path.exists(path):
        return False
    meta = get_meta(ds_id)
    meta["name"] = str(new_name).strip() or meta.get("name", "Tanpa nama")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return True
