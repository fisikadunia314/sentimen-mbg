FROM python:3.12-slim

# Agar log langsung keluar & tidak bikin file .pyc
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Install dependensi dulu (agar layer cache dipakai ulang kalau kode berubah)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin kode aplikasi
COPY . .

# Fly.io (dan platform lain) membaca port dari environment $PORT
EXPOSE 8080

# sh -c agar ${PORT} di-expand oleh shell
CMD ["sh", "-c", "gunicorn wsgi:app --bind 0.0.0.0:${PORT} --workers 1 --timeout 300"]
