"""Crawling data Instagram — post & komentar publik via Instaloader.

PERINGATAN (jujur):
  * API resmi Instagram HANYA membuka data akun sendiri (business/creator).
    Untuk konten publik orang lain, aplikasi ini memakai Instaloader yang
    mengakses halaman publik — ini wilayah abu-abu (berbeda dengan ToS IG).
  * Instagram agresif memblokir akses (CAPTCHA / limit). Jika diblokir,
    tunggulah beberapa jam. Disarankan memakai AKUN KEDUA (bukan akun utama)
    melalui INSTAGRAM_USERNAME / INSTAGRAM_PASSWORD di .env.
  * Gunakan untuk volume kecil (riset/akademik), bukan scraping massal.

Tanpa kredensial: mencoba akses anonim (sering dibatasi, terutama hashtag).
"""
import pandas as pd

MAX_TOTAL_COMMENTS = 20_000
MAX_COMMENTS_PER_POST = 100


class InstagramCredsError(ValueError):
    """Akses Instagram diblokir / login gagal."""


def _to_int(v) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def _source_to_df(query: str, mode: str, meta: dict) -> pd.DataFrame:
    """Petakan sumber (hashtag/profil) ke skema kolom 'video' aplikasi."""
    cols = ["video_id", "title", "channel", "published_at",
            "view_count", "like_count", "comment_count", "url"]
    if mode == "hashtag":
        title = f"#{query}"
        url = f"https://www.instagram.com/explore/tags/{query}/"
    else:
        title = f"@{query}"
        url = f"https://www.instagram.com/{query}/"
    return pd.DataFrame([{
        "video_id": f"ig_{mode}_{query}",
        "title": title,
        "channel": "Instagram",
        "published_at": "",
        "view_count": _to_int(meta.get("media_count")),
        "like_count": 0,
        "comment_count": _to_int(meta.get("comments_fetched")),
        "url": url,
    }], columns=cols)


def _comments_to_df(comments: list) -> pd.DataFrame:
    cols = ["comment_id", "parent_id", "type", "video_id", "author",
            "comment_text", "published_at", "like_count", "reply_count"]
    rows = []
    for cm in comments:
        rows.append({
            "comment_id": str(cm.get("comment_id")),
            "parent_id": cm.get("parent_id"),
            "type": cm["type"],
            "video_id": cm["video_id"],
            "author": cm.get("author") or "anonim",
            "comment_text": cm.get("comment_text") or "",
            "published_at": cm.get("published_at") or "",
            "like_count": _to_int(cm.get("likes")),
            "reply_count": 0,
        })
    return pd.DataFrame(rows, columns=cols)


IG_LOGIN_MSG = (
    "Instagram menolak akses anonim (login_required). Sejak 2024 Instagram "
    "mewajibkan LOGIN untuk membaca hashtag/profil publik, jadi kredensial "
    "SEKARANG WAJIB: isi INSTAGRAM_USERNAME & INSTAGRAM_PASSWORD di .env "
    "(pakai AKUN KEDUA, bukan akun utama), lalu jalankan ulang aplikasi."
)


def _ig_error(e):
    """Terjemahkan error instaloader jadi pesan yang bisa ditindaklanjuti."""
    msg = str(e).lower()
    if "login_required" in msg or "login required" in msg or "401" in msg:
        return IG_LOGIN_MSG
    if "not found" in msg or "404" in msg or "nonexistent" in msg:
        return "Tidak ditemukan — periksa ejaan hashtag/username."
    if "checkpoint" in msg or "challenge_required" in msg or "apc=" in msg:
        return ("AKUN TERKUNCI SEMENTARA (checkpoint) — Instagram tidak mengenali "
                "login ini sebagai kamu. INI BUKAN SALAH PASSWORD. Langkahnya:\n"
                "  1) Buka Instagram di browser HP/laptop, login akun tersebut.\n"
                "  2) Selesaikan verifikasi (kode email/SMS, 'It was me', atau ganti "
                "password bila diminta).\n"
                "  3) JANGAN jalankan crawl berulang-ulang — tiap percobaan membuat "
                "blokir makin lama.\n"
                "  4) Tunggu 24–72 jam, lalu coba lagi 1x.\n"
                "Jika tetap gagal: pakai akun lain yang sudah lama aktif & pernah "
                "login di HP (bukan akun baru), atau pakai menu Impor Data manual.")
    if "429" in msg or "too many" in msg or "rate" in msg or "wait" in msg:
        return ("Kena rate limit Instagram. Tunggu 1–3 jam, kurangi jumlah post/"
                "komentar, atau pakai akun kedua.")
    return f"Gagal mengakses Instagram: {e}"


def _build_context(username: str, password: str):
    try:
        import instaloader
    except ImportError:
        raise InstagramCredsError(
            "Paket 'instaloader' belum terinstall. Jalankan di terminal:\n"
            "    pip install instaloader\n"
            "lalu restart aplikasi."
        )
    # Instaloader 4.x tidak punya parameter `loglevel`; atur lewat logging.
    import logging
    logging.getLogger("instaloader").setLevel(logging.ERROR)
    L = instaloader.Instaloader(
        quiet=True,
        download_pictures=False,
        download_videos=False,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=False,
        max_connection_attempts=2,
        request_timeout=30,
    )
    if username and password:
        try:
            L.context.login(username, password)
        except Exception as e:
            raise InstagramCredsError(
                "Login Instagram gagal. " + _ig_error(e)
            )
    return L


def _comment_date(comment):
    """PostComment di instaloader 4.x hanya punya created_at_utc (UTC)."""
    dt = getattr(comment, "created_at_utc", None)
    if dt is None:
        dt = getattr(comment, "date_local", None) or getattr(comment, "date_utc", None)
    try:
        return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else ""
    except Exception:
        return ""


def _fetch_nodes(L, mode: str, query: str, max_posts: int):
    """Return (meta_dict, list_of_post_nodes)."""
    import instaloader
    if mode == "hashtag":
        try:
            hashtag = instaloader.Hashtag.from_name(L.context, query)
        except Exception as e:
            raise InstagramCredsError(f"Hashtag '#{query}': {_ig_error(e)}")
        media_count = 0
        try:
            media_count = hashtag.media_count
        except Exception:
            media_count = 0
        nodes = []
        for post in hashtag.get_posts():
            nodes.append(post)
            if len(nodes) >= max_posts:
                break
        return {"media_count": media_count}, nodes
    else:
        try:
            profile = instaloader.Profile.from_username(L.context, query)
        except Exception as e:
            raise InstagramCredsError(f"Profil '@{query}': {_ig_error(e)}")
        nodes = []
        for post in profile.get_posts():
            nodes.append(post)
            if len(nodes) >= max_posts:
                break
        return {"media_count": profile.mediacount}, nodes


def crawl_instagram_data(username: str, password: str, mode: str, query: str,
                         max_posts: int, max_comments: int,
                         videos_csv: str, comments_csv: str):
    """Pipeline crawling post & komentar Instagram (hashtag atau profil).

    Return (df_source, df_comments) dengan skema kolom identik dengan
    crawler YouTube/Telegram sehingga pipeline lanjutan tidak diubah.
    """
    mode = (mode or "hashtag").lower()
    if mode not in ("hashtag", "profile"):
        mode = "hashtag"
    query = (query or "").strip().lstrip("#@")
    if not query:
        raise InstagramCredsError("Isi kata kunci / nama profil Instagram.")
    max_posts = max(1, min(int(max_posts), 20))
    max_comments = max(1, min(int(max_comments), MAX_COMMENTS_PER_POST))

    try:
        L = _build_context((username or "").strip(), (password or "").strip())
    except InstagramCredsError:
        raise
    except Exception as e:
        raise InstagramCredsError(f"Gagal menyiapkan akses Instagram: {e}")

    try:
        meta, nodes = _fetch_nodes(L, mode, query, max_posts)
    except InstagramCredsError:
        raise
    except Exception as e:
        msg = str(e).lower()
        if "login" in msg or "2fa" in msg or "challenge" in msg or "rate" in msg or "block" in msg:
            raise InstagramCredsError(
                "Instagram membatasi akses (CAPTCHA/rate limit). "
                "Coba lagi beberapa jam kemudian, atau isi kredensial AKUN KEDUA "
                "di .env (INSTAGRAM_USERNAME / INSTAGRAM_PASSWORD)."
            )
        raise InstagramCredsError(f"Gagal mengambil data Instagram: {e}")

    all_comments = []
    comments_fetched = 0
    seen = set()
    for post in nodes:
        if len(all_comments) >= MAX_TOTAL_COMMENTS:
            break
        video_id = f"ig_{getattr(post, 'shortcode', None) or getattr(post, 'id', 0)}"
        try:
            for comment in post.get_comments():
                if comments_fetched >= max_comments:
                    break
                cid = getattr(comment, "id", None) or getattr(comment, "pk", None)
                if cid in seen:
                    continue
                seen.add(cid)
                owner = comment.owner.username if getattr(comment, "owner", None) else "anonim"
                body = (comment.text or "").strip()
                if not body:
                    continue
                all_comments.append({
                    "comment_id": cid,
                    "parent_id": None,
                    "type": "comment",
                    "video_id": video_id,
                    "author": owner,
                    "comment_text": body,
                    "published_at": _comment_date(comment),
                    "likes": _to_int(getattr(comment, "likes", 0)),
                })
                comments_fetched += 1
        except Exception:
            continue  # post terkunci/hapus — lewati

    meta["comments_fetched"] = comments_fetched
    df_source = _source_to_df(query, mode, meta)
    df_comments = _comments_to_df(all_comments)
    df_source.to_csv(videos_csv, index=False, encoding="utf-8-sig")
    df_comments.to_csv(comments_csv, index=False, encoding="utf-8-sig")
    return df_source, df_comments
