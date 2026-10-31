"""Crawling data Reddit — Reddit Data API v1 resmi (OAuth2 script flow).

Kuota gratis: 100 request/menit per token. Modul ini memasang rate-limit
(±90 request/menit) dan retry dengan backoff saat 429/5xx.

Kredensial (isi di .env):
    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USERNAME, REDDIT_PASSWORD

Panduan pembuatan app: README bagian "Sumber Data: Reddit".
"""
import base64
import time

import pandas as pd
import requests

REDDIT_API = "https://www.reddit.com"
# User-Agent wajib deskriptif (aturan Reddit)
_USER_AGENT = "linux:sentimen-mbg:v1.0 (aplikasi web analisis sentimen)"

MAX_TOTAL_COMMENTS = 20_000      # batas total agar CSV & kuota tetap sehat
REQUEST_INTERVAL = 0.65          # detik antar request (≈92 req/menit < kuota 100)

_token_cache = {"token": None, "expires_at": 0.0}


class RedditCredsError(ValueError):
    """Kredensial Reddit salah / tidak lengkap / token ditolak."""


# ---------------------------------------------------------------- tokens
def _basic_auth(client_id: str, client_secret: str) -> str:
    raw = f"{client_id}:{client_secret}".encode()
    return "Basic " + base64.b64encode(raw).decode()


def _get_access_token(creds: dict) -> str:
    """Ambil/ambil-dari-cache token OAuth (script flow, TTL 1 jam)."""
    now = time.time()
    if _token_cache["token"] and _token_cache["expires_at"] > now + 30:
        return _token_cache["token"]
    resp = requests.post(
        f"{REDDIT_API}/api/v1/access_token",
        headers={
            "Authorization": _basic_auth(creds["client_id"], creds["client_secret"]),
            "User-Agent": _USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={
            "grant_type": "password",
            "username": creds["username"],
            "password": creds["password"],
        },
        timeout=30,
    )
    if resp.status_code == 401:
        raise RedditCredsError(
            "Reddit menolak kredensial (401). Periksa REDDIT_CLIENT_ID / "
            "REDDIT_CLIENT_SECRET / REDDIT_USERNAME / REDDIT_PASSWORD di .env."
        )
    if resp.status_code != 200:
        raise RedditCredsError(f"Reddit OAuth gagal (HTTP {resp.status_code}).")
    data = resp.json()
    _token_cache["token"] = data["access_token"]
    _token_cache["expires_at"] = now + int(data.get("expires_in", 3600))
    return _token_cache["token"]


# ---------------------------------------------------------------- request
def _get_json(path: str, params: dict, creds: dict, retries: int = 4):
    """GET JSON dengan rate-limit + retry (429/5xx/network)."""
    last_err = None
    for attempt in range(retries):
        time.sleep(REQUEST_INTERVAL)
        try:
            resp = requests.get(
                f"{REDDIT_API}{path}",
                params=params,
                headers={
                    "Authorization": f"Bearer {_get_access_token(creds)}",
                    "User-Agent": _USER_AGENT,
                },
                timeout=30,
            )
        except requests.RequestException as e:
            last_err = e
            time.sleep(2 ** attempt)
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            last_err = ValueError(f"HTTP {resp.status_code}")
            time.sleep(2 ** attempt + 1)
            continue
        if resp.status_code == 403:
            raise RedditCredsError(
                "Reddit API menolak akses (403). Token/akun bermasalah — "
                "coba buat ulang app di reddit.com/prefs/apps."
            )
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"Gagal mengambil data Reddit setelah {retries} percobaan: {last_err}")


# ---------------------------------------------------------------- data
def search_posts(creds: dict, query: str, subreddit: str = "all", limit: int = 25) -> list:
    """Cari post (link) berdasarkan kata kunci pada satu subreddit / seluruh Reddit."""
    data = _get_json(
        "/search.json",
        {
            "q": query,
            "subreddit": (subreddit or "all").strip().removeprefix("r/") or "all",
            "limit": max(1, min(int(limit), 100)),
            "sort": "relevance",
            "type": "link",
            "raw_json": 1,
        },
        creds,
    )
    posts = []
    for child in data.get("data", {}).get("children", []):
        d = child.get("data", {})
        if not d or d.get("stickied") or d.get("over_18") or not d.get("id"):
            continue
        posts.append({
            "post_id": d["id"],
            "title": d.get("title", ""),
            "subreddit": d.get("subreddit", ""),
            "score": d.get("score", 0),
            "num_comments": d.get("num_comments", 0),
            "permalink": d.get("permalink", ""),
            "created_utc": d.get("created_utc", 0),
        })
    return posts


def _walk_comments(children: list, parent_id, out: list):
    for child in children:
        d = (child or {}).get("data", {})
        if not d or not d.get("id"):
            continue
        body = d.get("body") or ""
        if body in ("[removed]", "[deleted]"):
            continue
        replies = d.get("replies")
        n_replies = 0
        if isinstance(replies, dict):
            n_replies = len(replies.get("data", {}).get("children", []))
        out.append({
            "comment_id": d["id"],
            "parent_id": parent_id,
            "type": "comment" if parent_id is None else "reply",
            "author": d.get("author") or "[deleted]",
            "comment_text": body,
            "score": d.get("score", 0),
            "created_utc": d.get("created_utc", 0),
            "reply_count": n_replies,
        })
        if isinstance(replies, dict):
            _walk_comments(replies.get("data", {}).get("children", []), d["id"], out)


def get_post_comments(creds: dict, permalink: str, limit: int = 100) -> list:
    """Ambil komentar + reply sebuah post (1 request, maksimal 500 komentar)."""
    data = _get_json(
        f"{permalink}.json",
        {"limit": max(1, min(int(limit), 500)), "sort": "confidence", "raw_json": 1},
        creds,
    )
    children = []
    if isinstance(data, list):
        # [post_dict, comments_dict]
        if len(data) == 2 and isinstance(data[1], dict):
            children = data[1].get("data", {}).get("children", [])
    elif isinstance(data, dict):
        children = data.get("data", {}).get("children", [])
    out = []
    _walk_comments(children, None, out)
    return out


def _posts_to_df(posts: list) -> pd.DataFrame:
    """Petakan post Reddit ke skema kolom 'video' yang dipakai aplikasi."""
    cols = ["video_id", "title", "channel", "published_at",
            "view_count", "like_count", "comment_count", "url"]
    if not posts:
        return pd.DataFrame(columns=cols)
    rows, seen = [], set()
    for p in posts:
        if p["post_id"] in seen:
            continue
        seen.add(p["post_id"])
        created = int(p.get("created_utc") or 0)
        rows.append({
            "video_id": p["post_id"],
            "title": p.get("title", ""),
            "channel": f"r/{p.get('subreddit', '')}",
            "published_at": time.strftime("%Y-%m-%d", time.gmtime(created)) if created else "",
            "view_count": int(p.get("score") or 0),   # skor upvote (ditampilkan 'Views / Skor')
            "like_count": int(p.get("score") or 0),
            "comment_count": int(p.get("num_comments") or 0),
            "url": f"{REDDIT_API}{p.get('permalink', '')}",
        })
    return pd.DataFrame(rows, columns=cols)


def _comments_to_df(comments: list) -> pd.DataFrame:
    """Petakan komentar Reddit ke skema kolom komentar aplikasi (kolom sama dgn crawler YouTube)."""
    cols = ["comment_id", "parent_id", "type", "video_id", "author",
            "comment_text", "published_at", "like_count", "reply_count"]
    rows = []
    for c in comments:
        created = int(c.get("created_utc") or 0)
        rows.append({
            "comment_id": c["comment_id"],
            "parent_id": c["parent_id"],
            "type": c["type"],
            "video_id": c["video_id"],
            "author": c["author"],
            "comment_text": c["comment_text"],
            "published_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(created)) if created else "",
            "like_count": int(c.get("score") or 0),
            "reply_count": int(c.get("reply_count") or 0),
        })
    return pd.DataFrame(rows, columns=cols)


def crawl_reddit_data(creds: dict, queries: list, subreddit: str,
                      max_posts: int, max_comments_per_post: int,
                      videos_csv: str, comments_csv: str):
    """Pipeline crawling Reddit berdasarkan kata kunci.

    Return (df_posts, df_comments) dengan skema kolom yang sama persis
    dengan hasil crawler YouTube, sehingga pipeline lanjutan
    (preprocess → leksikon → model → laporan) tidak perlu diubah.
    """
    missing = [k for k in ("client_id", "client_secret", "username", "password")
               if not creds.get(k)]
    if missing:
        raise RedditCredsError(
            "Kredensial Reddit belum lengkap (hilang: " + ", ".join(missing) + "). "
            "Isi REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USERNAME, "
            "REDDIT_PASSWORD di .env — panduan di README bagian 'Sumber Data: Reddit'."
        )

    max_posts = max(1, min(int(max_posts), 25))
    max_comments = max(10, min(int(max_comments_per_post), 500))
    sub = (subreddit or "all").strip().removeprefix("r/") or "all"

    all_posts, all_comments = [], []
    seen_comments = set()
    for q in queries:
        posts = search_posts(creds, q, sub, limit=max_posts)
        for p in posts:
            all_posts.append(p)
            if len(all_comments) >= MAX_TOTAL_COMMENTS:
                break
            try:
                cmts = get_post_comments(creds, p["permalink"], limit=max_comments)
            except RedditCredsError:
                raise
            except Exception:
                continue  # post rusak/dihapus — lewati, lanjut post berikutnya
            for c in cmts:
                if c["comment_id"] in seen_comments:
                    continue
                seen_comments.add(c["comment_id"])
                c["video_id"] = p["post_id"]
                all_comments.append(c)
        if len(all_comments) >= MAX_TOTAL_COMMENTS:
            break

    df_posts = _posts_to_df(all_posts)
    df_comments = _comments_to_df(all_comments)
    df_posts.to_csv(videos_csv, index=False, encoding="utf-8-sig")
    df_comments.to_csv(comments_csv, index=False, encoding="utf-8-sig")
    return df_posts, df_comments
