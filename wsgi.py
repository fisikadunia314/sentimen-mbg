"""Entry point produksi (gunicorn) — dipakai saat deploy ke Render/Heroku/VPS.

Lokal tetap pakai:  python app.py
Produksi pakai:     gunicorn wsgi:app --bind 0.0.0.0:$PORT --workers 1
"""
from app import app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
