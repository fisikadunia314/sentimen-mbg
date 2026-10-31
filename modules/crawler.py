"""Modul crawling komentar YouTube menggunakan YouTube Data API v3 (v2).

Penambahan:
- Retry sederhana dengan backoff untuk error 5xx/429.
- Fetching reply PARALEL (ThreadPoolExecutor) — ini bagian paling lama saat
  komentar ribuan, karena tiap thread dengan reply memicu request tambahan.
  Total kuota API tetap sama, yang berkurang adalah waktu tunggu.
"""
import os
import re
import socket
import time
import threading
import pandas as pd
from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor, as_completed
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}

# Error JARINGAN (bukan HTTP error): Windows WinError 10060/10054/10051,
# macOS/Linux ConnectionResetError, DNS gagal, SSL timeout, dsb.
NETWORK_ERRORS = (OSError, socket.timeout, TimeoutError, ConnectionError,
                  socket.gaierror)

REQUEST_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "60"))   # detik per request
NET_RETRIES = int(os.getenv("NET_RETRIES", "3"))         # percobaan ulang


class CrawlNetworkError(RuntimeError):
    """Gagal menghubungi server Google/YouTube (masalah jaringan, bukan kode)."""


def friendly_network_message(e, host: str = "youtube.googleapis.com") -> str:
    """Terjemahkan error jaringan mentah menjadi instruksi yang bisa dilakukan."""
    raw = str(e)
    low = raw.lower()
    if "10060" in raw or "timed out" in low or "timeout" in low:
        cause = "koneksi ke server YouTube/Google kehabisan waktu (timeout)"
    elif "10054" in raw or "10051" in raw or "reset" in low or "aborted" in low:
        cause = "koneksi diputus di tengah jalan oleh jaringan/firewall"
    elif "getaddrinfo" in low or "name or service" in low or "11001" in raw:
        cause = "nama domain tidak bisa diterjemahkan (masalah DNS)"
    elif "ssl" in low or "certificate" in low:
        cause = "koneksi aman (SSL) ditolak — biasanya karena proxy/VPN/firewall kantor"
    else:
        cause = "server tidak bisa dihubungi"
    return (
        f"Gagal menghubungi {host}: {cause}.\n"
        "Ini masalah JARINGAN di komputer/ISP kamu, bukan kesalahan data. "
        "Coba berurutan:\n"
        "  1) Pastikan internet jalan — buka https://www.youtube.com di browser.\n"
        "  2) Matikan VPN/proxy/DNS-pribadi sementara, lalu ulangi.\n"
        "  3) Ganti DNS ke 8.8.8.8 / 1.1.1.1 (kadang ISP memblokir googleapis.com).\n"
        "  4) Kalau di kantor/kampus: jaringan mereka mungkin memblokir API Google.\n"
        "  5) Pakai jaringan lain (mis. hotspot HP) untuk mengujinya.\n"
        f"Detail teknis: {raw}"
    )

# Klien API terpisah per thread (klien googleapiclient tidak thread-safe)
_tls = threading.local()


def _http_client():
    """httplib2 client dengan TIMEOUT + dukungan proxy dari variabel lingkungan.

    Tanpa timeout eksplisit, request bisa menggantung sangat lama; dengan
    proxy (HTTPS_PROXY/HTTP_PROXY) aplikasi tetap jalan di jaringan kantor.
    """
    import httplib2
    proxy_info = None
    try:
        proxy_info = httplib2.proxy_info_from_environment()
    except Exception:
        proxy_info = None
    if proxy_info:
        return httplib2.Http(timeout=REQUEST_TIMEOUT, proxy_info=proxy_info)
    return httplib2.Http(timeout=REQUEST_TIMEOUT)


def _thread_client(api_key: str):
    client = getattr(_tls, "client", None)
    if client is None:
        client = build("youtube", "v3", developerKey=api_key, http=_http_client())
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
    return build("youtube", "v3", developerKey=api_key, http=_http_client())


def _execute_with_retry(request_obj, retries: int = None, backoff: float = 2.0):
    """Eksekusi request dengan retry untuk error sementara.

    Ditangani:
      * HTTP 5xx / 429  -> dicoba ulang (server sibuk)
      * Error JARINGAN  -> dicoba ulang (WinError 10060, connection reset, DNS)
    Setelah semua percobaan habis, error jaringan diubah menjadi
    CrawlNetworkError dengan pesan yang bisa ditindaklanjuti pengguna.
    """
    if retries is None:
        retries = NET_RETRIES
    last_net_error = None
    for attempt in range(retries + 1):
        try:
            return request_obj.execute()
        except HttpError as e:
            status = e.resp.status if e.resp is not None else 0
            if status in _RETRYABLE_STATUS and attempt < retries:
                time.sleep(backoff * (attempt + 1))
                continue
            raise
        except NETWORK_ERRORS as e:
            # Error jaringan: jeda makin lama, lalu coba lagi
            last_net_error = e
            if attempt < retries:
                time.sleep(backoff * (2 ** attempt))   # 2s, 4s, 8s
                continue
            raise CrawlNetworkError(friendly_network_message(e)) from e
    if last_net_error is not None:
        raise CrawlNetworkError(friendly_network_message(last_net_error))


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
        # Kuota habis / video menonaktifkan komentar -> hentikan, pakai yang sudah ada
        print(f"[WARN] get_comments_with_replies {video_id}: {e}")
        if not all_comments and e.resp is not None and e.resp.status in (403, 404):
            raise
    except NETWORK_ERRORS as e:
        # Jaringan putus di tengah pengambilan ribuan komentar:
        # JANGAN buang hasil yang sudah didapat.
        if not all_comments:
            raise CrawlNetworkError(friendly_network_message(e)) from e
        print(f"[WARN] jaringan terputus setelah {len(all_comments)} komentar: {e}")

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


def crawl_search_data(api_key: str, queries: list, max_videos: int,
                   max_pages: int, videos_csv: str, comments_csv: str):
    """Pipeline crawling berdasarkan kata kunci pencarian YouTube."""
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
