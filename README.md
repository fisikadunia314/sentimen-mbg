# SentimenYT — Analisis Sentimen Komentar YouTube (MBG)

Aplikasi web **Flask + Tailwind CSS** (dashboard bergaya admin) untuk tugas praktikum
analisis sentimen terhadap kebijakan **MBG (Makan Bergizi Gratis)** dari komentar YouTube.

## Fitur

| Fitur | Keterangan |
|---|---|
| 🎬 **Analisis Video** (fitur utama) | Tempel URL video YouTube → komentar & reply di-crawl via YouTube Data API v3, diproses, diberi label, dan divisualisasikan (distribusi sentimen, komentar vs reply, detail per komentar). Jika model sudah dilatih, hasil prediksi Naïve Bayes & SVM + tingkat kesesuaian dengan leksikon juga ditampilkan. |
| 📥 Crawling dataset MBG | Crawl komentar dari beberapa video hasil pencarian kata kunci. |
| 🧹 Preprocessing | Cleaning → normalisasi kata alay → stopword (negasi dipertahankan) → stemming Sastrawi. |
| 🏷️ Labeling leksikon | Skor sentimen berbasis leksikon Bahasa Indonesia dengan penanganan **negasi (jendela 3 token, mendukung negasi ganda)**, **bigram** ("luar biasa", "terima kasih"), dan **intensifier** ("banget", "sangat"). |
| 🤖 Training | Naïve Bayes & SVM (Linear) dengan fitur TF-IDF 1–2 gram. Sudah ada guard untuk dataset kecil. |
| 📊 Evaluasi | Accuracy, macro precision/recall/F1, confusion matrix (chart), laporan per kelas. |
| ✍️ Prediksi komentar baru | Uji komentar bebas — bandingkan hasil NB, SVM, dan leksikon. |
| 📄 Laporan PDF (ala makalah) | **Isi identitas penulis dulu (nama, NIM, prodi, dosen)** — tersimpan otomatis. Unduh laporan otomatis: cover (dengan identitas) → pendahuluan → metodologi/cara kerja (6 tahap + tabel teknik) → hasil analisis video (data video, tabel & grafik distribusi, komentar vs reply, sampel komentar) → **evaluasi hasil** (metrik model, confusion matrix, performa per kelas, kesesuaian leksikon-model, interpretasi) → kesimpulan & saran. Bisa langsung dari video terakhir, atau tempel URL video baru (akan dianalisis dulu). |
| 🎨 UI | Tailwind CSS (CSS sudah dikompilasi, **jalan offline**), sidebar ala admin panel, responsif mobile. |

## Cara Menjalankan

```bash
# 1) Clone / ekstrak proyek, lalu masuk ke foldernya
cd sentimen-web

# 2) Buat virtual environment (opsional tapi disarankan)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3) Install dependensi
pip install -r requirements.txt

# 4) Siapkan konfigurasi
cp .env.example .env            # Windows: copy .env.example .env
#    lalu isi YOUTUBE_API_KEY di file .env

# 5) Jalankan aplikasi
python app.py
```

Buka http://localhost:5000

### Mendapatkan YouTube API Key

1. Buka [Google Cloud Console](https://console.cloud.google.com/).
2. Buat project → **APIs & Services → Library** → aktifkan **YouTube Data API v3**.
3. **Credentials → Create Credentials → API Key** → salin ke `.env`.

> Kuota default 10.000 unit/hari. Mengambil 1 halaman komentar ≈ 1 unit; setiap thread
> yang memiliki reply menambah permintaan lagi (max 2 halaman reply per thread).

## Alur Kerja

```
Crawling (YouTube Data API v3)
   └─ data/komentar_youtube_mbg.csv + video_sumber_mbg.csv
        └─ Preprocessing & Labeling (Sastrawi + leksikon)
             └─ kolom: clean, normalized, no_stopword, stemmed, norm_tokens, score, label
                  └─ Training (NB & SVM + TF-IDF)
                       └─ models/*.pkl + metrics.json
                            └─ Evaluasi + Dashboard + Prediksi
```

## Struktur Proyek

```
sentimen-web/
├── app.py                  # Route Flask (dashboard, analisis URL, pipeline, API JSON)
├── config.py               # Konfigurasi (path, kuota crawl, env)
├── requirements.txt
├── .env.example            # Template konfigurasi lingkungan
├── modules/
│   ├── crawler.py          # YouTube Data API v3 (retry 5xx/429, reply paralel)
│   ├── preprocess.py       # Pipeline teks (Sastrawi, cache per kata unik)
│   ├── lexicon.py          # Leksikon sentimen + aturan negasi/bigram/intensifier
│   ├── labeling.py         # Labeling berbasis leksikon
│   ├── modeling.py         # Training NB & SVM (guard dataset kecil)
│   ├── evaluator.py        # Prediksi + visualisasi (model di-cache)
│   └── report.py           # Generator laporan PDF (reportlab, gaya makalah)
├── templates/              # Template Jinja (Tailwind, sidebar admin)
├── static/
│   ├── css/tailwind.min.css  # Tailwind hasil kompilasi (offline)
│   └── charts/           # Chart PNG hasil generate (dibuat saat aplikasi berjalan)
├── data/                   # CSV hasil crawling (dibuat otomatis)
└── models/                 # File .pkl + metrics.json (dibuat saat training)
```

## API JSON (sederhana)

- `GET /api/metrics` → metrik evaluasi model
- `GET /api/distribution` → distribusi label komentar

## Catatan & Keterbatasan

- Label ground truth dihasilkan **labeling leksikon** (bukan anotasi manusia);
  metrik evaluasi mengukur kesesuaian model terhadap leksikon. Interpretasi hasil
  sebaiknya disertai validasi manual beberapa sampel.
- Leksikon terbatas (~100 kata) dan dapat diperluas di `modules/lexicon.py`.
- Data komentar terbatas oleh kuota API dan batas halaman yang dipilih.
