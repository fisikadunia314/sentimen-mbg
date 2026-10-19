"""Modul crawling komentar YouTube menggunakan YouTube Data API v3 (v2).

Penambahan:
- Retry sederhana dengan backoff untuk error 5xx/429.
- Fetching reply PARALEL (ThreadPoolExecutor) — ini bagian paling lama saat
  komentar ribuan, karena tiap thread dengan reply memicu request tambahan.
  Total kuota API tetap sama, yang berkurang adalah waktu tunggu.
"""
import os
import re
import time
import threading
import pandas as pd
from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor, as_completed
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}

# Klien API terpisah per thread (klien googleapiclient tidak thread-safe)
_tls = threading.local()


def _thread_client(api_key: str):
    client = getattr(_tls, "client", None)
    if client is None:
        client = build("youtube", "v3", developerKey=api_key)
        _tls.client = client
    return client


def _to_int(value) -> int:
    """Paksa nilai statistik menjadi int (dapat berupa string/None/NaN)."""
    if value is None:
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def build_youtube_client(api_key: str):
    return build("youtube", "v3", developerKey=api_key)


def _execute_with_retry(request_obj, retries: int = 2, backoff: float = 2.0):
    """Eksekusi request dengan retry untuk error sementara (5xx/429)."""
    for attempt in range(retries + 1):
        try:
            return request_obj.execute()
        except HttpError as e:
            status = e.resp.status if e.resp is not None else 0
            if status in _RETRYABLE_STATUS and attempt < retries:
                time.sleep(backoff * (attempt + 1))
                continue
            raise


# ==================== EKSTRAK VIDEO ID ====================
def extract_video_id(url: str) -> str:
    """Ekstrak video ID dari berbagai format URL YouTube."""
    if not url:
        return ""
    url = url.strip()

    # Format youtu.be/VIDEOID
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0].split("&")[0]

    # Format youtube.com/watch?v=VIDEOID
    parsed = urlparse(url)
    if parsed.hostname and "youtube.com" in parsed.hostname:
        qs = parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0]
        # Format youtube.com/embed/VIDEOID atau /shorts/VIDEOID
        m = re.search(r"/(embed|shorts|v)/([A-Za-z0-9_-]{11})", parsed.path)
        if m:
            return m.group(2)

    # Jika input hanya video ID (11 karakter)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url

    return ""


def get_video_info(youtube, video_id: str) -> dict:
    """Ambil metadata video."""
    try:
        resp = _execute_with_retry(
            youtube.videos().list(part="snippet,statistics", id=video_id)
        )
        if not resp.get("items"):
            return {}
        item = resp["items"][0]
        s, st = item["snippet"], item.get("statistics") or {}
        return {
            "video_id": video_id,
            "title": s.get("title", "") or "",
            "channel": s.get("channelTitle", "") or "",
            "published_at": s.get("publishedAt", "") or "",
            "view_count": _to_int(st.get("viewCount")),
            "like_count": _to_int(st.get("likeCount")),
            "comment_count": _to_int(st.get("commentCount")),
            "url": f"https://www.youtube.com/watch?v={video_id}",
        }
    except HttpError as e:
        print(f"[ERROR] get_video_info: {e}")
        return {}


# ==================== AMBIL REPLY ====================
def get_replies(youtube, parent_comment_id: str, max_pages: int = 2) -> list:
    """Mengambil reply dari sebuah komentar top-level."""
    replies = []
    next_page_token = None
    page = 0
    try:
        while page < max_pages:
            resp = _execute_with_retry(
                youtube.comments().list(
                    part="snippet",
                    parentId=parent_comment_id,
                    maxResults=100,
                    pageToken=next_page_token,
                    textFormat="plainText",
                )
            )

            for item in resp.get("items", []):
                s = item["snippet"]
                replies.append({
                    "comment_id": item["id"],
                    "parent_id": parent_comment_id,
                    "type": "reply",
                    "author": s.get("authorDisplayName", ""),
                    "comment_text": s.get("textDisplay", ""),
                    "published_at": s.get("publishedAt", ""),
                    "like_count": _to_int(s.get("likeCount")),
                })
            next_page_token = resp.get("nextPageToken")
            page += 1
            if not next_page_token:
                break
    except HttpError as e:
        print(f"[WARN] get_replies {parent_comment_id}: {e}")
    return replies


def get_comments_with_replies(youtube, video_id: str, max_pages: int = 3,
                              include_replies: bool = True,
                              max_reply_pages: int = 2,
                              api_key: str = None, max_workers: int = 6) -> list:
    """Ambil komentar top-level + seluruh reply.

    Halaman top-level diambil berurutan (butuh pageToken berantai),
    lalu seluruh reply di-fetch PARALEL dengan `max_workers` thread.
    """
    all_comments = []
    parent_ids = []
    next_page_token = None
    page = 0
    try:
        while page < max_pages:
            response = _execute_with_retry(
                youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=100,
                    pageToken=next_page_token,
                    textFormat="plainText",
                    order="relevance",
                )
            )

            for item in response.get("items", []):
                top = item["snippet"]["topLevelComment"]["snippet"]
                parent_id = item["id"]

                # Komentar top-level
                all_comments.append({
                    "comment_id": parent_id,
                    "parent_id": None,
                    "type": "comment",
                    "video_id": video_id,
                    "author": top.get("authorDisplayName", ""),
                    "comment_text": top.get("textDisplay", ""),
                    "published_at": top.get("publishedAt", ""),
                    "like_count": _to_int(top.get("likeCount")),
                    "reply_count": _to_int(item["snippet"].get("totalReplyCount")),
                })

                # Kumpulkan thread yang punya reply (diambil paralel di bawah)
                if include_replies and _to_int(item["snippet"].get("totalReplyCount")) > 0:
                    parent_ids.append(parent_id)

            next_page_token = response.get("nextPageToken")
            page += 1
            if not next_page_token:
                break
    except HttpError as e:
        print(f"[WARN] get_comments_with_replies {video_id}: {e}")

    # ===== Fetch reply PARALEL =====
    if include_replies and parent_ids and api_key:
        def _fetch(pid):
            return get_replies(_thread_client(api_key), pid, max_pages=max_reply_pages)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_fetch, pid) for pid in parent_ids]
            for fut in as_completed(futures):
                try:
                    for r in fut.result():
                        r["video_id"] = video_id
                        r["reply_count"] = 0
                        all_comments.append(r)
                except Exception as e:
                    print(f"[WARN] fetch reply gagal: {e}")

    return all_comments


# ==================== CRAWL DARI SATU URL ====================
def crawl_from_url(api_key: str, url: str, max_pages: int = 5,
                   include_replies: bool = True):
    """Crawl komentar + reply dari SATU URL YouTube."""
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY belum di-set. Salin .env.example menjadi .env lalu isi API key-nya.")

    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("URL YouTube tidak valid. Contoh: https://www.youtube.com/watch?v=XXXX")

    youtube = build_youtube_client(api_key)

    # Metadata video
    video_info = get_video_info(youtube, video_id)
    if not video_info:
        raise ValueError("Video tidak ditemukan atau tidak dapat diakses.")

    # Komentar + reply
    comments = get_comments_with_replies(
        youtube, video_id,
        max_pages=max_pages,
        include_replies=include_replies,
        api_key=api_key,
    )

    df_videos = pd.DataFrame([video_info])
    df_comments = pd.DataFrame(comments)

    if not df_comments.empty:
        df_comments = df_comments.drop_duplicates(subset=["comment_id"])

    return df_videos, df_comments


# ==================== CRAWL BERDASARKAN KATA KUNCI ====================
def search_videos(youtube, query: str, max_results: int = 5):
    """Mencari video berdasarkan kata kunci."""
    videos = []
    try:
        response = _execute_with_retry(
            youtube.search().list(
                q=query, part="id,snippet", maxResults=max_results,
                type="video", relevanceLanguage="id",
            )
        )
        for item in response.get("items", []):
            if item["id"].get("kind") != "youtube#video":
                continue
            vid = item["id"]["videoId"]
            snippet = item["snippet"]
            stats_resp = _execute_with_retry(
                youtube.videos().list(part="statistics", id=vid)
            )
            stats = stats_resp["items"][0]["statistics"] if stats_resp["items"] else {}
            videos.append({
                "video_id": vid, "title": snippet.get("title", ""),
                "channel": snippet.get("channelTitle", ""),
                "published_at": snippet.get("publishedAt", ""),
                "view_count": _to_int(stats.get("viewCount")),
                "like_count": _to_int(stats.get("likeCount")),
                "comment_count": _to_int(stats.get("commentCount")),
                "url": f"https://www.youtube.com/watch?v={vid}",
            })
    except HttpError as e:
        print(f"[ERROR] search_videos: {e}")
    return videos


def crawl_mbg_data(api_key: str, queries: list, max_videos: int,
                   max_pages: int, videos_csv: str, comments_csv: str):
    """Pipeline crawling berdasarkan kata kunci (dataset utama MBG)."""
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY belum di-set. Salin .env.example menjadi .env lalu isi API key-nya.")

    youtube = build_youtube_client(api_key)
    all_videos, all_comments = [], []
    for q in queries:
        vids = search_videos(youtube, q, max_results=max_videos)
        all_videos.extend(vids)
        for v in vids:
            cmts = get_comments_with_replies(
                youtube, v["video_id"], max_pages=max_pages, include_replies=True,
                api_key=api_key,
            )
            all_comments.extend(cmts)

    df_videos = pd.DataFrame(all_videos)
    df_comments = pd.DataFrame(all_comments)
    if not df_videos.empty:
        df_videos = df_videos.drop_duplicates(subset=["video_id"])
    if not df_comments.empty:
        df_comments = df_comments.drop_duplicates(subset=["comment_id"])

    df_videos.to_csv(videos_csv, index=False, encoding="utf-8-sig")
    df_comments.to_csv(comments_csv, index=False, encoding="utf-8-sig")
    return df_videos, df_comments
