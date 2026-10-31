"""Login Telegram SEKALI (membuat file sesi untuk fitur Crawling → Telegram).

Cara pakai:
    python setup_telegram.py

Butuh TELEGRAM_API_ID & TELEGRAM_API_HASH sudah terisi di .env
(dapatkan gratis di https://my.telegram.org → "API development tools").
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv

load_dotenv()

api_id = os.getenv("TELEGRAM_API_ID", "").strip()
api_hash = os.getenv("TELEGRAM_API_HASH", "").strip()
if not api_id or not api_hash:
    print("TELEGRAM_API_ID / TELEGRAM_API_HASH belum diisi di .env.")
    print("Dapatkan gratis: buka https://my.telegram.org → Login → API development tools")
    sys.exit(1)

try:
    from telethon.sync import TelegramClient
except ImportError:
    print("=" * 62)
    print("Paket 'telethon' belum terinstall.")
    print("Jalankan dulu (di terminal yang sama / venv yang sama):")
    print()
    print("    pip install telethon")
    print()
    print("atau install semua kebutuhan sekaligus:")
    print()
    print("    pip install -r requirements.txt")
    print("=" * 62)
    sys.exit(1)

SESSION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "telegram_session.session")
os.makedirs(os.path.dirname(SESSION), exist_ok=True)

client = TelegramClient(SESSION, api_id, api_hash)
with client:
    if not client.is_user_authorized():
        print("Login baru dibutuhkan (sekali saja).")
        phone = input("Nomor HP dengan kode negara (mis. +6281234567890): ").strip()
        client.send_code_request(phone)
        code = input("Kode login yang muncul di aplikasi Telegram kamu: ").strip()
        try:
            client.sign_in(phone, code)
        except Exception:
            pw = input("Dipaksa password 2FA? Ketik di sini (kosongkan jika tidak): ").strip()
            client.sign_in(phone, code, password=pw or None)
    me = client.get_me()
    print(f"\nLogin OK: {me.first_name or me.username or me.id}")
    print(f"File sesi tersimpan: {SESSION}")
    print("Sekarang fitur Crawling → Telegram siap dipakai.")
