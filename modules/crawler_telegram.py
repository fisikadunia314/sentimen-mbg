"""Crawling data Telegram — channel publik via Telethon (user session).

Cara pakai:
  1) Dapatkan api_id & api_hash GRATIS di https://my.telegram.org
     (Log in dengan nomor HP → "API development tools" → isi nama app → Create).
  2) Isi TELEGRAM_API_ID & TELEGRAM_API_HASH di .env
  3) Jalankan SEKALI:  python setup_telegram.py   (login via kode dari Telegram,
     membuat file sesi data/telegram_session.session)
  4) Setelah itu fitur Crawling → sumber Telegram bisa dipakai tanpa login lagi.

Catatan: memakai API user session resmi Telegram untuk penggunaan pribadi —
kompatibel dengan ToS Telegram untuk riset/akademik berskala kecil.
"""
import os

import pandas as pd

try:
    from telethon.sync import TelegramClient
    TELEGRAM_AVAILABLE = True
except ImportError:          # paket belum diinstall -> app tetap bisa jalan
    TelegramClient = None
    TELEGRAM_AVAILABLE = False

MAX_TOTAL_MESSAGES = 20_000   # pagar total agar CSV & kuota tetap sehat


class TelegramCredsError(ValueError):
    """Kredensial Telegram salah / sesi belum dibuat."""


def _parse_channel(raw: str) -> str:
    """Normalisasi input channel: 'https://t.me/xxx' / '@xxx' / 'xxx' -> 'xxx'."""
    s = (raw or "").strip()
    if s.startswith("https://t.me/"):
        s = s[len("https://t.me/"):]
    elif s.startswith("http://t.me/"):
        s = s[len("http://t.me/"):]
    s = s.strip("/").split("/")[0]
    return s.removeprefix("@")


def _channel_url(name: str) -> str:
    return f"https://t.me/s/{name}"


def _entity_name(entity):
    """Nama publik entity (username atau id) untuk get_entity."""
    if entity is None:
        return None
    if getattr(entity, "username", None):
        return entity.username
    if getattr(entity, "title", None):
        return entity.title
    return getattr(entity, "id", None)


def _to_int(v) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def _channels_to_df(channels: list) -> pd.DataFrame:
    """Petakan channel ke skema kolom 'video' aplikasi."""
    cols = ["video_id", "title", "channel", "published_at",
            "view_count", "like_count", "comment_count", "url"]
    rows = []
    for ch in channels:
        rows.append({
            "video_id": f"tg_{ch['name']}",
            "title": ch.get("title") or ch["name"],
            "channel": f"@{ch['name']}" if ch.get("name") else "Telegram",
            "published_at": "",
            "view_count": _to_int(ch.get("subscribers")),
            "like_count": 0,
            "comment_count": _to_int(ch.get("messages_fetched")),
            "url": _channel_url(ch["name"]),
        })
    return pd.DataFrame(rows, columns=cols)


def _messages_to_df(messages: list) -> pd.DataFrame:
    """Petakan pesan Telegram ke skema kolom komentar aplikasi."""
    cols = ["comment_id", "parent_id", "type", "video_id", "author",
            "comment_text", "published_at", "like_count", "reply_count"]
    rows = []
    for m in messages:
        rows.append({
            "comment_id": str(m["comment_id"]),
            "parent_id": m.get("parent_id"),
            "type": m["type"],
            "video_id": m["video_id"],
            "author": m.get("author") or "channel",
            "comment_text": m.get("comment_text") or "",
            "published_at": m.get("published_at") or "",
            "like_count": _to_int(m.get("views")),
            "reply_count": _to_int(m.get("reply_count")),
        })
    return pd.DataFrame(rows, columns=cols)


def _iter_messages(client, entity, limit: int):
    return client.iter_messages(entity, limit=limit)


def crawl_telegram_data(api_id, api_hash, session_file: str, channels: list,
                        max_messages: int, include_replies: bool,
                        videos_csv: str, comments_csv: str):
    if not TELEGRAM_AVAILABLE:
        raise TelegramCredsError(
            "Paket 'telethon' belum terinstall. Jalankan di terminal:\n"
            "    pip install telethon\n"
            "lalu restart aplikasi."
        )
    """Pipeline crawling channel publik Telegram berdasarkan nama channel.

    `channels`: list string (nama/username/link channel publik).
    Return (df_channels, df_messages) dengan skema kolom identik dengan
    crawler YouTube/Instagram sehingga pipeline lanjutan tidak perlu diubah.
    """
    if not api_id or not api_hash:
        raise TelegramCredsError(
            "TELEGRAM_API_ID / TELEGRAM_API_HASH belum diisi di .env. "
            "Dapatkan gratis di https://my.telegram.org — panduan di README "
            "bagian 'Sumber Data: Telegram'."
        )
    if not os.path.exists(session_file):
        raise TelegramCredsError(
            "Sesi Telegram belum dibuat. Jalankan sekali:  python setup_telegram.py  "
            "(butuh kode login dari aplikasi Telegram kamu)."
        )

    max_messages = max(10, min(int(max_messages), 500))

    client = TelegramClient(session_file, str(api_id), api_hash)
    all_channels, all_messages = [], []
    seen = set()
    try:
        client.connect()
        if not client.is_user_authorized():
            raise TelegramCredsError(
                "Sesi Telegram tidak valid / kedaluwarsa. Jalankan lagi:  "
                "python setup_telegram.py"
            )
        for raw in channels:
            name = _parse_channel(raw)
            if not name:
                continue
            try:
                entity = client.get_entity(name)
            except Exception as e:
                raise TelegramCredsError(
                    f"Channel '{name}' tidak ditemukan (harus channel PUBLIK). "
                    f"Detail: {e}"
                )
            title = getattr(entity, "title", "") or name
            subs = getattr(entity, "members_count", 0) or 0
            ch_key = {"name": _entity_name(entity) or name, "title": title,
                      "subscribers": subs, "messages_fetched": 0}
            all_channels.append(ch_key)
            video_id = f"tg_{ch_key['name']}"

            for msg in _iter_messages(client, entity, limit=max_messages):
                if len(all_messages) >= MAX_TOTAL_MESSAGES:
                    break
                text = (msg.message or "").strip()
                if not text:
                    continue  # media tanpa teks — lewati (pipeline butuh teks)
                author = ""
                sender = msg.sender
                if sender is not None:
                    author = (getattr(sender, "username", None)
                              or getattr(sender, "first_name", None)
                              or getattr(sender, "title", None)) or ""
                n_replies = 0
                reply_msgs = []
                if include_replies and getattr(msg, "replies", None) is not None:
                    try:
                        reply_list = client.get_messages(msg.replies, limit=50)
                        for rm in reply_list:
                            if not rm or not (rm.message or "").strip():
                                continue
                            rs = rm.sender
                            ra = ""
                            if rs is not None:
                                ra = (getattr(rs, "username", None)
                                      or getattr(rs, "first_name", None)) or ""
                            reply_msgs.append({
                                "comment_id": rm.id,
                                "parent_id": msg.id,
                                "type": "reply",
                                "video_id": video_id,
                                "author": ra,
                                "comment_text": rm.message.strip(),
                                "published_at": rm.date.strftime("%Y-%m-%d %H:%M:%S") if rm.date else "",
                                "views": _to_int(rm.views),
                                "reply_count": 0,
                            })
                            seen.add(("r", video_id, rm.id))
                        n_replies = len(reply_msgs)
                    except Exception:
                        pass  # reply tidak tersedia — lanjut
                mid = msg.id
                if (mid, video_id) in seen:
                    continue
                seen.add((mid, video_id))
                all_messages.append({
                    "comment_id": mid,
                    "parent_id": None,
                    "type": "comment",
                    "video_id": video_id,
                    "author": author,
                    "comment_text": text,
                    "published_at": msg.date.strftime("%Y-%m-%d %H:%M:%S") if msg.date else "",
                    "views": _to_int(msg.views),
                    "reply_count": n_replies,
                })
                all_messages.extend(reply_msgs)
                ch_key["messages_fetched"] += 1
    finally:
        client.disconnect()

    df_channels = _channels_to_df(all_channels)
    df_messages = _messages_to_df(all_messages)
    df_channels.to_csv(videos_csv, index=False, encoding="utf-8-sig")
    df_messages.to_csv(comments_csv, index=False, encoding="utf-8-sig")
    return df_channels, df_messages
