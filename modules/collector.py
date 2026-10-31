"""Koleksi data dari SATU LINK (YouTube / Instagram / Telegram).

Alur: pengguna menempel satu URL -> platform dideteksi otomatis -> komentar
diambil -> dibuat DataFrame dengan skema kolom yang SAMA seperti crawler lain
(`video_id,title,...,url` untuk sumber; `comment_id,...,comment_text,...` untuk
komentar), sehingga pipeline preprocessing/training/laporan tidak perlu diubah.

Tidak ada "AI yang scraping sendiri": yang menentukan cara mengambil data
adalah POLA LINK (regex) di bawah. Untuk platform yang tidak didukung, fungsi
akan menolak dengan pesan jelas — bukan mencoba menembus proteksi situs.
"""
import re

import pandas as pd

VIDEO_COLS = ["video_id", "title", "channel", "published_at",
              "view_count", "like_count", "comment_count", "url"]
COMMENT_COLS = ["comment_id", "parent_id", "type", "video_id", "author",
                "comment_text", "published_at", "like_count", "reply_count"]


class CollectError(ValueError):
    """Error yang aman ditampilkan ke pengguna (tanpa traceback)."""


# ============================== DETEKSI PLATFORM ==============================
IG_RE = re.compile(r"instagram\.com/(?:[^/]+/)?(p|reel|reels|tv)/([A-Za-z0-9_-]+)", re.I)
TG_RE = re.compile(r"(?:t\.me|telegram\.me)/(?:s/)?([A-Za-z0-9_]{3,})/(\d+)", re.I)
YE_RE = re.compile(r"(?:youtube\.com/(?:watch\?[^#]*v=|shorts/|live/|embed/)|youtu\.be/)"
                   r"([A-Za-z0-9_-]{6,})", re.I)


def detect_source(url: str) -> str:
    """Tentukan platform dari pola link: 'youtube' | 'instagram' | 'telegram'."""
    u = (url or "").strip()
    if not u:
        raise CollectError("Tempel dulu link-nya.")
    if IG_RE.search(u):
        return "instagram"
    if TG_RE.search(u):
        return "telegram"
    if YE_RE.search(u):
        return "youtube"
    if re.search(r"instagram\.com", u, re.I):
        raise CollectError(
            "Link Instagram harus berupa link POST, mis. "
            "https://www.instagram.com/p/ABC123/ atau /reel/ABC123/ — "
            "bukan link profil/hashtag."
        )
    if re.search(r"t\.me|telegram\.me", u, re.I):
        raise CollectError(
            "Link Telegram harus menunjuk satu PESAN, mis. "
            "https://t.me/namaChannel/12345"
        )
    if re.search(r"youtube\.com|youtu\.be", u, re.I):
        raise CollectError(
            "Link YouTube tidak dikenali. Contoh: "
            "https://www.youtube.com/watch?v=XXXXXXXXXXX"
        )
    raise CollectError(
        "Platform belum didukung. Yang bisa: link video YouTube, link post "
        "Instagram (/p/ atau /reel/), dan link pesan Telegram (t.me/channel/id). "
        "Untuk platform lain, pakai menu Impor Data (tempel komentar manual)."
    )


def _to_int(v):
    try:
        return int(str(v).replace(".", "").replace(",", ""))
    except Exception:
        return 0


def _dt(dt):
    try:
        return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else ""
    except Exception:
        return ""


# ================================= YOUTUBE =================================
def _collect_youtube(url: str, max_comments: int, include_replies: bool, api_key: str):
    from modules.crawler import crawl_from_url
    max_pages = max(1, min(20, (max_comments // 100) + 1))
    dfv, dfc = crawl_from_url(api_key=api_key, url=url, max_pages=max_pages,
                              include_replies=include_replies)
    if dfc is not None and not dfc.empty:
        if max_comments:
            dfc = dfc.head(max_comments)
        dfc = dfc[[c for c in COMMENT_COLS if c in dfc.columns]]
    else:
        dfc = pd.DataFrame(columns=COMMENT_COLS)
    if dfv is not None and not dfv.empty:
        dfv = dfv[[c for c in VIDEO_COLS if c in dfv.columns]]
    else:
        dfv = pd.DataFrame(columns=VIDEO_COLS)
    return dfv, dfc


# ================================ INSTAGRAM ================================
def _collect_instagram(url: str, max_comments: int, username: str, password: str):
    from modules.crawler_instagram import _build_context, _ig_error
    m = IG_RE.search(url)
    if not m:
        raise CollectError("Link Instagram tidak dikenali.")
    kind, shortcode = m.group(1).lower(), m.group(2)

    try:
        import instaloader
    except ImportError:
        raise CollectError(
            "Paket 'instaloader' belum terinstall. Jalankan: pip install instaloader"
        )

    L = _build_context((username or "").strip(), (password or "").strip())
    try:
        post = instaloader.Post.from_shortcode(L.context, shortcode)
    except Exception as e:
        raise CollectError(f"Gagal membuka post Instagram: {_ig_error(e)}")

    owner = getattr(getattr(post, "owner", None), "username", "") or "instagram"
    caption = (getattr(post, "caption", "") or "").strip().replace("\n", " ")
    title = (caption[:90] + "…") if len(caption) > 90 else (caption or f"Post @{owner}")
    dfv = pd.DataFrame([{
        "video_id": f"ig_{shortcode}",
        "title": title,
        "channel": f"@{owner}",
        "published_at": _dt(getattr(post, "date_utc", None)),
        "view_count": _to_int(getattr(post, "video_view_count", 0)),
        "like_count": _to_int(getattr(post, "likes", 0)),
        "comment_count": _to_int(getattr(post, "comments", 0)),
        "url": f"https://www.instagram.com/{kind}/{shortcode}/",
    }], columns=VIDEO_COLS)

    rows, seen = [], set()
    try:
        for c in post.get_comments():
            if max_comments and len(rows) >= max_comments:
                break
            cid = getattr(c, "id", None) or getattr(c, "pk", None)
            if cid in seen:
                continue
            seen.add(cid)
            body = (getattr(c, "text", "") or "").strip()
            if not body:
                continue
            rows.append({
                "comment_id": str(cid),
                "parent_id": None,
                "type": "comment",
                "video_id": f"ig_{shortcode}",
                "author": getattr(getattr(c, "owner", None), "username", "") or "anonim",
                "comment_text": body,
                "published_at": _dt(getattr(c, "created_at_utc", None)),
                "like_count": _to_int(getattr(c, "likes", 0)),
                "reply_count": 0,
            })
    except Exception as e:
        if not rows:
            raise CollectError(f"Gagal mengambil komentar Instagram: {_ig_error(e)}")

    return dfv, pd.DataFrame(rows, columns=COMMENT_COLS)


# ================================= TELEGRAM ================================
def _collect_telegram(url: str, max_comments: int, include_replies: bool,
                      api_id, api_hash, session_file):
    from modules.crawler_telegram import (TelegramClient, TELEGRAM_AVAILABLE,
                                          _parse_channel, _to_int as _tg_int)
    if not TELEGRAM_AVAILABLE:
        raise CollectError(
            "Paket 'telethon' belum terinstall. Jalankan: pip install telethon"
        )
    import os
    if not (api_id and api_hash):
        raise CollectError(
            "TELEGRAM_API_ID / TELEGRAM_API_HASH belum diisi di .env "
            "(gratis di my.telegram.org)."
        )
    if not os.path.exists(session_file):
        raise CollectError(
            "Sesi Telegram belum ada. Jalankan sekali: python setup_telegram.py"
        )

    m = TG_RE.search(url)
    if not m:
        raise CollectError("Link Telegram tidak dikenali.")
    channel, msg_id = _parse_channel(m.group(1)), int(m.group(2))

    client = TelegramClient(session_file, str(api_id), api_hash)
    try:
        client.connect()
        if not client.is_user_authorized():
            raise CollectError(
                "Sesi Telegram tidak valid. Jalankan lagi: python setup_telegram.py"
            )
        try:
            entity = client.get_entity(channel)
        except Exception as e:
            raise CollectError(
                f"Channel '{channel}' tidak bisa dibuka (harus publik & sudah kamu "
                f"ikuti). Detail: {e}"
            )
        try:
            msg = client.get_messages(entity, ids=msg_id)
        except Exception as e:
            raise CollectError(f"Pesan {msg_id} gagal diambil: {e}")
        if not msg:
            raise CollectError(f"Pesan {msg_id} tidak ditemukan di '{channel}'.")

        ch_name = getattr(entity, "username", None) or channel
        ch_title = getattr(entity, "title", "") or ch_name
        video_id = f"tg_{ch_name}_{msg_id}"
        body = (msg.message or "").strip()
        title = (body[:90] + "…") if len(body) > 90 else (body or f"Pesan {msg_id}")
        dfv = pd.DataFrame([{
            "video_id": video_id,
            "title": title,
            "channel": ch_title,
            "published_at": _dt(getattr(msg, "date", None)),
            "view_count": _to_int(getattr(msg, "views", 0)),
            "like_count": 0,
            "comment_count": 0,
            "url": f"https://t.me/{ch_name}/{msg_id}",
        }], columns=VIDEO_COLS)

        rows = []
        if body:
            rows.append({
                "comment_id": str(msg_id), "parent_id": None, "type": "comment",
                "video_id": video_id, "author": ch_title,
                "comment_text": body,
                "published_at": _dt(getattr(msg, "date", None)),
                "like_count": 0, "reply_count": 0,
            })
        if include_replies and getattr(msg, "replies", None) is not None:
            try:
                for rm in client.get_messages(msg.replies, limit=max_comments or 100):
                    if not rm:
                        continue
                    txt = (rm.message or "").strip()
                    if not txt:
                        continue
                    sender = getattr(rm, "sender", None)
                    author = ((getattr(sender, "username", None)
                               or getattr(sender, "first_name", None)) if sender else "") or "anonim"
                    rows.append({
                        "comment_id": str(rm.id), "parent_id": str(msg_id),
                        "type": "reply", "video_id": video_id, "author": author,
                        "comment_text": txt,
                        "published_at": _dt(getattr(rm, "date", None)),
                        "like_count": 0, "reply_count": 0,
                    })
                    if max_comments and len(rows) >= max_comments:
                        break
            except Exception:
                pass
        if not rows:
            raise CollectError(
                "Pesan itu tidak punya teks/komentar yang bisa dianalisis "
                "(mungkin kiriman media tanpa caption)."
            )
        return dfv, pd.DataFrame(rows, columns=COMMENT_COLS)
    finally:
        try:
            client.disconnect()
        except Exception:
            pass


# ================================== DISPATCH ===============================
def collect_from_link(url: str, max_comments: int = 100,
                      include_replies: bool = True,
                      youtube_api_key: str = "",
                      ig_username: str = "", ig_password: str = "",
                      tg_api_id="", tg_api_hash="", tg_session=""):
    """Ambil komentar dari satu link. Return (source: str, df_videos, df_comments)."""
    source = detect_source(url)
    url = url.strip()
    if source == "youtube":
        if not youtube_api_key:
            raise CollectError(
                "YOUTUBE_API_KEY belum diisi di .env — dibutuhkan untuk link YouTube."
            )
        dfv, dfc = _collect_youtube(url, max_comments, include_replies, youtube_api_key)
    elif source == "instagram":
        dfv, dfc = _collect_instagram(url, max_comments, ig_username, ig_password)
    else:
        dfv, dfc = _collect_telegram(url, max_comments, include_replies,
                                     tg_api_id, tg_api_hash, tg_session)

    if dfc is None or dfc.empty:
        raise CollectError("Tidak ada komentar yang berhasil diambil dari link itu.")
    return source, dfv, dfc
