# SentimenKu — Analisis Sentimen Komentar Media Sosial

Aplikasi web **Flask + Tailwind CSS** (dashboard bergaya admin) untuk analisis
sentimen komentar **media sosial** secara umum: **YouTube**, **Telegram**, dan
**Instagram**. Pipeline (crawling → preprocessing → labeling → training →
evaluasi → laporan PDF) seragam untuk semua sumber data.

## Fitur

| Fitur | Keterangan |
|---|---|
| 🎬 **Analisis Video** (fitur utama) | Tempel URL video YouTube → komentar & reply di-crawl via YouTube Data API v3, diproses, diberi label, dan divisualisasikan (distribusi sentimen, komentar vs reply, detail per komentar). Jika model sudah dilatih, hasil prediksi Naïve Bayes & SVM + tingkat kesesuaian dengan leksikon juga ditampilkan. |
| 🔗 Data Koleksi | Tempel satu link (YouTube/Instagram/Telegram) → komentar diambil otomatis & disimpan jadi CSV. |
| 🎭 Deteksi Sarkasme | Komentar sindiran dikenali dari petunjuk (frasa, emoji, konflik polaritas) lalu polaritas labelnya dibalik. |
| 🗂️ Dataset | Setiap koleksi tersimpan terpisah — pilih, gabung, hapus, atau reset semua dari satu halaman. |
| 🩺 Tes Koneksi | Uji DNS/HTTPS ke setiap layanan yang dipakai aplikasi — membedakan masalah jaringan vs data. |
| 📥 Impor data manual | Tempel komentar (1/baris) atau unggah CSV saat crawling otomatis diblokir — masuk pipeline yang sama. |
| 📥 Crawling dataset | Crawl komentar dari **YouTube, Telegram, atau Instagram** (pilih sumber di form). Kata kunci/channel bebas, pengaturan volume per sumber, hasil pencarian tampil sebagai tabel sumber (judul bisa diklik untuk validasi). Semua sumber masuk pipeline yang sama. |
| 🧹 Preprocessing | 8 tahap: case folding → cleaning (URL/HTML/emoji/angka/mention/hashtag) → tokenisasi → normalisasi alay → stopword (negasi dipertahankan) → stemming Sastrawi → labeling leksikon. Ada panel **Kualitas Hasil Preprocessing** untuk memeriksa sisa kata. |
| 🏷️ Labeling leksikon | Skor sentimen berbasis leksikon Bahasa Indonesia dengan penanganan **negasi (jendela 3 token, mendukung negasi ganda)**, **bigram** ("luar biasa", "terima kasih"), dan **intensifier** ("banget", "sangat"). |
| 🤖 Training | Pilih model: **Naïve Bayes + SVM** (TF-IDF), **LSTM/BiLSTM**, **GRU/BiGRU**, dan **BERT/IndoBERT** (fine-tuning). Semua dinilai dengan split 80/20 dan dibandingkan di halaman Evaluasi. |
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

### Sumber Data: Telegram (opsional, gratis, tanpa approval)

Data: **pesan channel publik** + reply thread (mis. channel berita).

**Mendapatkan kredensial (±2 menit, gratis):**
1. Buka [my.telegram.org](https://my.telegram.org) → login dengan nomor HP.
2. **API development tools** → isi nama app bebas → **Create application**
   → catat `api_id` dan `api_hash`.
3. Isi `TELEGRAM_API_ID` & `TELEGRAM_API_HASH` di `.env`.
4. Jalankan **sekali** di terminal (folder proyek):
   ```bash
   python setup_telegram.py
   ```
   Ikuti permintaan nomor HP + kode login dari aplikasi Telegram kamu.
   File sesi `data/telegram_session.session` terbentuk → tidak perlu login lagi.

**Memakai:** menu Crawling → sumber **Telegram** → isi channel (1 per baris,
mis. `medcomnews`, `@kompascom`, atau `t.me/xxx`) → jumlah pesan per channel →
opsi sertakan reply. Channel harus **publik** dan sudah kamu ikuti di aplikasi
Telegram (agar nama channel bisa di-resolve).

### Sumber Data: Instagram (opsional — wilayah abu-abu)

Data: **post + komentar** dari hashtag publik atau profil.

> **PERINGATAN JUJUR:** API resmi Instagram hanya membuka data *akun sendiri*.
> Modul ini memakai **Instaloader** (akses halaman publik) — berbeda dengan ToS
> Instagram dan bisa kena CAPTCHA/blokir. **Gunakan AKUN KEDUA** (bukan akun
> utama) dan volume kecil (riset/akademik), bukan scraping massal.

**Kredensial WAJIB (per 2024+):** Instagram kini menolak akses anonim
(`login_required`) untuk hashtag maupun profil publik. Isi
`INSTAGRAM_USERNAME` & `INSTAGRAM_PASSWORD` di `.env` memakai **akun kedua**
(jangan akun utama — risiko CAPTCHA/blokir), lalu restart aplikasi.

**Jika muncul `Checkpoint required` / CAPTCHA** — akun kamu sedang ditandai
mencurigakan oleh Instagram (bukan salah password):
1. Buka Instagram di browser HP/laptop, login dengan akun itu, selesaikan
   verifikasi (kode email/SMS, "It was me", atau ubah password bila diminta).
2. **Jangan** menjalankan crawl berulang-ulang — tiap percobaan memperpanjang blokir.
3. Tunggu 24–72 jam, lalu coba **sekali**. Akun baru/akun tanpa aktivitas paling
   sering kena ini; akun lama yang biasa login di HP jauh lebih stabil.
4. Kalau masih gagal → pakai menu **Impor Data** (lihat bawah).

**Memakai:** menu Crawling → sumber **Instagram** → pilih hashtag/profil →
isi kuncinya (mis. `kebijakan` atau `medcomnews`) → batas post & komentar per post.

### Data Koleksi (tempel link → CSV)

Menu **Data Koleksi** menerima **satu link** dan menyimpannya sebagai dataset:

| Platform | Bentuk link yang diterima | Cara ambil |
|---|---|---|
| YouTube | `youtube.com/watch?v=…`, `youtu.be/…`, `youtube.com/shorts/…` | YouTube Data API v3 |
| Instagram | `instagram.com/p/<kode>/`, `/reel/<kode>/` | Instaloader (butuh login) |
| Telegram | `t.me/<channel>/<idPesan>` | Telethon (channel publik yang kamu ikuti) |

Platform dikenali otomatis dari **pola link**, lalu diambil lewat jalur yang sesuai
per platform — bukan "AI yang scraping sendiri". Opsi: batas jumlah komentar,
sertakan reply, dan **gabung ke data lama**. Hasil langsung tersimpan di
`data/komentar_media.csv` + `data/video_sumber.csv`, siap masuk Preprocessing.

Platform lain (TikTok/X/Facebook) sengaja tidak didukung karena tidak ada jalur
resmi — gunakan **Impor Data** manual.

### Impor Data Manual (jalan keluar resmi)

Menu **Impor Data** menerima komentar tanpa crawling sama sekali:

- **Tempel teks** — 1 komentar per baris (paling praktis untuk Instagram).
- **Unggah CSV** — berkolom `comment_text` (atau `komentar` / `text` / `isi`);
  file satu kolom juga diterima.
- Opsi **"tambahkan ke data yang sudah ada"** agar tidak menimpa hasil crawl sebelumnya.

Baris kosong dan duplikat otomatis dibuang, lalu data disimpan ke
`data/komentar_media.csv` dengan skema persis sama seperti hasil crawling —
jadi Preprocessing → Labeling → Training → Laporan PDF langsung jalan.

Cara manual yang aman untuk Instagram: buka post di browser → "Lihat semua
komentar" → blok & copy komentar → tempel di menu Impor Data. Ini legal
(membaca halaman publik sebagai pengguna biasa) dan tidak memicu blokir.

### Sumber yang TIDAK tersedia: Facebook

Facebook menutup akses API publik ke konten orang lain sejak lama — tidak ada
cara resmi (dan scraping-nya melanggar ToS serta sangat tidak stabil). Oleh
karena itu tidak disediakan dalam aplikasi ini.

## Bahasa & Model Deep Learning

### Tiga mode bahasa: Indonesia, English, **Campuran**

Bahasa dipilih di halaman **Preprocessing** dan **Training** (tersimpan di
`data/pengaturan.json`).

| | Bahasa Indonesia | English | Campuran (ID+EN) |
|---|---|---|---|
| Cleaning & normalisasi | Sastrawi + kamus alay ±130 entri | daftar singkatan Inggris (`dont`→`do not`, `pls`→`please`) | **per komentar**, sesuai bahasa terdeteksi |
| Stopword | Sastrawi + ±200 kata tambahan | daftar stopword Inggris (±200) | sesuai bahasa baris |
| Stemming | Sastrawi (kata sentimen/domain dilindungi) | **tidak di-stem** (morfologi ditangani model) | Sastrawi hanya untuk baris ID |
| Labeling | leksikon ID (±130 kata) | leksikon EN (±300 kata) | leksikon bahasa baris; jika skor 0 dicoba leksikon lainnya |
| Transformer | **IndoBERT** | **BERT** | **dua model**: IndoBERT untuk baris ID + BERT untuk baris EN, dipilih otomatis saat prediksi |

Ganti bahasa → **jalankan ulang Preprocessing**, karena label & token berbeda.

### Mode Campuran (dataset Indonesia + Inggris)

Berguna kalau komentar YouTube/Instagram-mu bercampur dua bahasa — umum pada
akun berita Indonesia.

Cara kerja:
1. **Deteksi bahasa per komentar** (`modules/lang_detect.py`) — berbasis kata
   penanda, imbuhan khas, dan kata sentimen dari leksikon. Tanpa dependensi
   eksternal, deterministik, dan bisa dijelaskan di laporan. Akurasi uji
   internal: 24/24 komentar campuran terdeteksi benar.
2. Setiap baris diproses dengan pipeline sesuai bahasanya (Sastrawi hanya untuk
   Indonesia), lalu diberi label memakai leksikon bahasa itu.
3. **Naive Bayes / SVM / LSTM / GRU**: satu model pada korpus campuran (token
   sudah sesuai bahasa masing-masing).
4. **BERT**: dua model dilatih terpisah — IndoBERT dari baris Indonesia, BERT
   dari baris Inggris. Saat prediksi, bahasa teks dideteksi dulu lalu model yang
   tepat dipakai. Metrik di halaman Evaluasi dihitung pada **test set gabungan**
   yang sama, jadi adil dibandingkan dengan model lain.

Catatan:
* Satu bahasa dilatih hanya jika subsetnya ≥ **10** komentar; bila hanya satu
  bahasa memenuhi syarat, model bahasa itu dipakai untuk semua teks.
* Halaman Preprocessing menampilkan **distribusi bahasa terdeteksi** di panel
  Kualitas Hasil Preprocessing.
* Halaman Prediksi menampilkan model mana yang dipakai
  (*model: IndoBERT* / *model: BERT*) untuk teks yang kamu masukkan.

> **Penting:** kalau data bercampur tapi mode bahasa dipilih **Indonesia**
> saja, komentar Inggris akan menjadi banyak **netral** (kata Inggris tidak ada
> di leksikon Indonesia). Itu bukan bug — pakai mode **Campuran**.

### Model deep learning (opsional)

LSTM/GRU dan BERT **tidak wajib**. Tanpa paketnya aplikasi tetap jalan penuh
dengan Naive Bayes + SVM.

```bash
pip install -r requirements-dl.txt
```

Lalu di halaman **Training** centang model yang diinginkan
(Naïve Bayes+SVM / LSTM / GRU / BERT).

| Model | Butuh | Waktu (CPU, ±500 komentar) | Catatan |
|---|---|---|---|
| Naïve Bayes + SVM | — | < 2 detik | baseline, paling stabil untuk data kecil |
| LSTM / BiLSTM | `torch` | ±10–60 detik | embedding dilatih dari korpus sendiri (tanpa unduhan) |
| GRU / BiGRU | `torch` | ±10–45 detik | lebih ringan dari LSTM |
| BERT / IndoBERT | `torch` + `transformers` | ±1–10 menit | unduh model 1x lalu di-cache |

**Model transformer default** (bisa diganti di `.env`):

```
BERT_MODEL_ID=indobenchmark/indobert-lite-base-p1   # ringan untuk CPU
BERT_MODEL_EN=distilbert-base-uncased
```

Versi lebih akurat (lebih berat): `indobenchmark/indobert-base-p1` dan
`bert-base-uncased`.

### Perbaikan aturan negasi (penting untuk validitas label)

Jendela negasi dipersempit menjadi **maksimal 2 token** dan **berhenti** saat
melewati kata bermuatan sentimen lain. Sebelumnya ("tidak setuju, mubazir uang
negara") menghasilkan skor 0 karena kata "tidak" ikut menegasi "mubazir";
sekarang benar **−2**. Regresi leksikon tetap sesuai rancangan:
`tidak setuju` −1, `tidak tidak setuju` +1, `enak banget` +2, `luar biasa` +1,
`buruk sekali` −2, `matur nuwun` 0.

### Catatan metodologis (penting untuk laporan)

Dengan dataset ratusan komentar, **BERT sering TIDAK mengalahkan TF-IDF+SVM**,
dan lebih mudah overfit. Itu bukan kegagalan — itu temuan yang layak dibahas:
model pra-latih butuh data jauh lebih banyak untuk unggul. Halaman Evaluasi
menampilkan tabel perbandingan semua model (accuracy/precision/recall/F1 macro)
supaya bisa dibandingkan secara langsung.

LSTM/GRU dan BERT memerlukan minimal **20** data berlabel; di bawah itu
aplikasi menolak dengan pesan jelas (Naive Bayes/SVM tetap bisa).

## Manajemen Dataset (setiap koleksi tersimpan sendiri)

Sebelumnya **semua koleksi saling menimpa** satu file kerja — akibatnya hasil
**Analisis Video** tertimpa oleh hasil **Crawling** (dan sebaliknya), sehingga
Preprocessing bisa memproses data yang salah. Sekarang:

| Aksi | Hasil |
|---|---|
| Analisis Video (1 link YT) | tersimpan sebagai dataset **"YouTube: judul video"** |
| Crawling kata kunci | dataset **"YouTube crawl: kata kunci"** |
| Crawling Telegram / Instagram | dataset sesuai channel/hashtag |
| Data Koleksi (link) | dataset **"Koleksi <platform>: <link>"** |
| Impor Data (paste/CSV) | dataset **"Impor: <nama sumber>"** |

Lokasi: `data/datasets/<id>/` berisi `meta.json`, `comments.csv`, `videos.csv`.
Dataset menyimpan **teks mentah** — preprocessing & labeling dijalankan saat
dataset dipilih, sehingga kamu bisa mengganti bahasa/mode kapan saja.

### Menu **Dataset**

- Tabel semua dataset: nama, sumber, jumlah komentar, waktu dibuat
- **Pakai** → jadikan data aktif lalu langsung ke Preprocessing
- **Hapus** → hapus satu dataset
- **Reset total (Zona Berbahaya)** → menghapus semua dataset + data kerja +
  hasil analisis + **model terlatih**, supaya benar-benar mulai dari nol

### Memilih data di Preprocessing

Di halaman **Preprocessing** ada kartu **"Pilih Dataset Sumber"**:

- centang **satu atau beberapa** dataset → digabung otomatis
- atau centang **"Gabung semua dataset"**
- gabungan otomatis menghapus duplikat (per `video_id` + `comment_id`, karena ID
  dari Telegram/Instagram bisa kembar antar sumber)
- tombol **Pakai Dataset Ini** menjadikannya data aktif, lalu klik
  **Jalankan Preprocessing**

## Deteksi Sarkasme (label dibalik)

Komentar sindiran adalah titik lemah analisis leksikon: kata-katanya positif,
maksudnya negatif.

| Komentar | Leksikon | Setelah deteksi sarkasme |
|---|---|---|
| `mantap betul program ini sampai rakyat makin susah` | 0 → netral ❌ | **−1 → negatif** ✔ |
| `hebat ya, katanya gratis tapi tagihan tetap datang` | +1 → positif ❌ | **−1 → negatif** ✔ |
| `luar biasa bagusnya sampai saya mengantri tiga jam` | +1 → positif ❌ | **−1 → negatif** ✔ |
| `terima kasih pak, sekarang malah makin mahal` | 0 → netral ❌ | **−1 → negatif** ✔ |
| `program ini bagus sekali saya setuju` | +4 → positif ✔ | tetap positif (tidak dianggap sarkasme) |

**Cara kerja** (`modules/sarcasm.py`, berbasis petunjuk — bukan model):
1. **Frasa sindiran** berbobot: `mantap betul`, `hebat ya`, `katanya`, `padahal`,
   `kenyataannya`, `dibilang`, `biar kelihatan bagus`, `ya kan`, `/s`, `so called`.
2. **Pola "sampai + akibat menyedihkan"**: `…sampai antri tiga jam`, `…sampai makin susah`.
3. **Emoji ironi**: 🙃 😏 🤡 👏 🙄 😒.
4. **Tanda baca**: `!!`, `...`, pujian di dalam tanda kutip (`"bagus"`).
5. **Konflik polaritas**: kata positif + negatif + penanda kontras dalam satu kalimat.

Skor ≥ 2 → dianggap sarkasme. Saat preprocessing, kolom yang dihasilkan:
`sarkasme`, `sarkasme_skor`, `sarkasme_petunjuk`, `score_leksikon`, `score`,
`sarkasme_dibalik`. Skor **asli tetap disimpan** (`score_leksikon`) agar bisa diaudit.

Toggle **"Tangani sarkasme (balik polaritas)"** ada di halaman Preprocessing
(default **aktif**). Bila dimatikan, sarkasme tetap dideteksi & ditandai, tetapi
label tidak dibalik — berguna untuk membandingkan hasil di laporan.
Halaman **Prediksi** juga menampilkan peringatan sarkasme beserta petunjuknya.

**Batas kemampuan (jujur untuk laporan):** deteksi ini **heuristik**, bukan model
terlatih. Akurasinya bergantung pada cakupan daftar petunjuk; sarkasme halus
tanpa penanda (mis. "ya bagus sih" saja) bisa lolos. Untuk tugas, sebutkan
keterbatasan ini dan tunjukkan contoh yang berhasil/gagal.

### Catatan teknis: warning `id2label` pada IndoBERT

Checkpoint `indobenchmark/indobert-lite-base-p1` aslinya dilatih untuk **5 kelas
sentimen**, sehingga muncul peringatan:

```
You passed num_labels=3 which is incompatible to the id2label map of length 5.
```

Ini **bukan error** — bobot klasifikasi lama memang dibuang dan diganti 3 kelas
kita. Sejak versi ini, `id2label`/`label2id` disetel eksplisit dan log
transformers diredam, jadi warning tersebut tidak lagi mengganggu. Bobot
`classifier.*` yang berstatus `MISSING` pada LOAD REPORT juga normal: itu layer
baru yang akan dilatih pada data kita.

## Masalah Jaringan (WinError 10060 / connection timeout)

Crawling **ribuan komentar** memerlukan ratusan request ke Google. Kalau satu
koneksi gagal, dulu seluruh proses berhenti dan hasil yang sudah didapat hilang.
Sekarang ditangani:

| Perbaikan | Efek |
|---|---|
| **Retry otomatis** untuk error jaringan (bukan hanya HTTP 5xx/429) | WinError 10060 / 10054 / DNS gagal dicoba ulang 3x dengan jeda 2s→4s→8s |
| **Timeout eksplisit** 60 detik per request | tidak menggantung selamanya |
| **Hasil parsial disimpan** | jaringan putus di komentar ke-800 → 800 komentar tetap masuk CSV, tidak hangus |
| **Dukungan proxy** dari variabel lingkungan | jalan di jaringan kantor/kampus yang memakai proxy |
| **Pesan error yang bisa ditindaklanjuti** | bukan `WinError 10060` mentah, tapi langkah perbaikannya |

### Menu "Tes Koneksi"

Buka menu **Tes Koneksi** di sidebar → **Jalankan Tes Koneksi**. Aplikasi menguji
DNS + HTTPS ke YouTube API, Google, HuggingFace, Instagram, Telegram, lalu
menampilkan tabel OK/GAGAL beserta waktu respons.

Cara membaca hasil:

| Hasil | Artinya | Tindakan |
|---|---|---|
| DNS gagal | internet/DNS bermasalah | ganti DNS ke 8.8.8.8 / 1.1.1.1 |
| DNS OK, HTTPS gagal | diblokir firewall/ISP/VPN | matikan VPN, pakai hotspot HP |
| YouTube gagal, Google OK | `googleapis.com` diblokir jaringan | pakai jaringan lain |
| Semua OK | jaringan sehat | gangguan sesaat — crawl ulang |

### Pengaturan di `.env`

```
HTTP_TIMEOUT=60        # naikkan ke 120 kalau internet lambat
NET_RETRIES=3          # percobaan ulang saat koneksi gagal
# HTTPS_PROXY=http://proxy.kantor:8080
```

## Tahap Preprocessing (detail)

Mengikuti materi *Text Mining & LDA* (Pertemuan 12), dengan tambahan aturan
khusus teks media sosial:

| # | Tahap | Yang dibersihkan |
|---|---|---|
| 1 | Case folding | `DATA Mining` → `data mining` |
| 2 | Cleaning | entitas HTML (`&amp; &quot; &nbsp;`), tag HTML (`<p>`, `<br/>`), URL (`http`, `www`, `t.co`, `bit.ly`), mention `@user`, hashtag `#tag`, angka, **emoji & seluruh karakter non-alfabet** |
| 3 | Tokenisasi | pemisahan per spasi |
| 4 | Normalisasi | kamus alay/singkatan ±120 entri (`gk`→`tidak`, `bgt`→`banget`, `sdh`→`sudah`, `jln`→`jalan`, `pmrth`→`pemerintah`, `tau`→`tahu`, …) + huruf berulang (`mantapppp`→`mantap`) |
| 5 | Stopword removal | Sastrawi + ±200 kata tambahan (partikel `sih/ya/doang`, sapaan `pak/kak/min`, sisa HTML `amp/quot/nbsp/br`, sampah `rt/rw/hp/rp`, tawa `wkwk/hehe/awok`). **Negasi & intensifier tidak dibuang.** |
| 6 | Stemming | Sastrawi; kata bermuatan sentimen (`setuju`, `gagal`) dan kata domain (`kebijakan`, `pemerintah`, `kesehatan`) **dilindungi** agar tidak rusak menjadi `tuju`/`bijak`/`perintah` |
| 7 | Output | `tokens` (untuk modeling) dan `norm_tokens` (untuk leksikon) |
| 8 | Labeling | leksikon + negasi + bigram + intensifier |

### Menambah stopword sendiri

Buat file `data/stopwords_custom.txt` (satu kata per baris, `#` untuk komentar),
lalu jalankan ulang Preprocessing. Cocok untuk membuang kata yang masih muncul
di panel **Kualitas Hasil Preprocessing** pada halaman Preprocessing
(panel itu menampilkan 30 kata paling sering + statistik token).

## Alur Kerja

```
Crawling (YouTube Data API v3 / Telegram / Instagram)
   └─ data/komentar_media.csv + video_sumber.csv
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

## Deploy ke Cloud (Render)

Aplikasi ini adalah web server Flask **berstate** (menyimpan model, data, dan grafik di disk
lokal) dengan proses crawling yang bisa berjalan beberapa menit — jadi **tidak cocok untuk
Vercel** (serverless, disk tidak persisten, timeout fungsi pendek). Platform yang tepat:
**Render.com** (free tier).

### Langkah
1. Push repo ini ke GitHub (lihat instruksi di atas; file `.env` sudah di-`.gitignore`
   jadi API key tidak ikut terupload).
2. Buat akun di [render.com](https://render.com) → **New → Blueprint** → pilih repo ini
   (Render otomatis membaca `render.yaml`).
   - Atau manual: **New → Web Service** → Build Command `pip install -r requirements.txt`,
     Start Command `gunicorn wsgi:app --bind 0.0.0.0:$PORT --workers 1 --timeout 300`.
3. Isi **Environment** di dashboard Render:
   - `YOUTUBE_API_KEY` = API key YouTube kamu
   - `SECRET_KEY` = string acak (mis. hasil `python -c "import secrets; print(secrets.token_hex(32))"`)
   - `FLASK_DEBUG` = `0`
4. **Deploy** → buka URL `https://<nama>.onrender.com`.

### Catatan free tier Render
- Instance **tidur setelah ±15 menit tidak aktif** → request pertama setelah jeda
  loading ±30–60 detik (normal, biarkan sampai muncul).
- Disk **tidak persisten** antar deploy: setelah redeploy, jalankan lagi
  *Analisis Video → Training Model* melalui UI (semua prosesnya memang by-design lewat UI).
- Crawling ribuan komentar bisa 1–5 menit; kalau request ter-cut, kurangi
  *Maks. Halaman Komentar* (mis. 5 → 3).

## Deploy ke Cloud (Fly.io)

Fly.io menjalankan aplikasi sebagai container Docker 24 jam (tidak tidur sembarangan
kecuali idle), region terdekat Indonesia: **Singapore (sin)** — sudah diset di `fly.toml`.

### Persiapan
1. Push repo ke GitHub (`.env` tidak ikut karena sudah di-`.gitignore`).
2. Install **flyctl** di Windows: `winget install fly.io.flyctl` (atau `scoop install flyctl`).
3. Buat akun di [fly.io](https://fly.io) → di terminal: `fly auth login`.

### Deploy
```
cd sentimen-web
fly launch          # deteksi Dockerfile otomatis; pilih nama app (mis. sentimenu);
                    # region: pilih SIN (Singapore); jawaban lain default (Enter)
fly secrets set YOUTUBE_API_KEY=... SECRET_KEY=...
fly deploy
```
Buka `https://<nama-app>.fly.dev`.

### Catatan
- Akun Fly.io perlu kartu kredit (plan Hobby). Mesin `shared-cpu-1x` 512MB sekitar
  beberapa dolar/bulan; ada kredit gratis untuk akun baru. `auto_stop_machines`
  mematikan mesin saat tidak ada pengunjung (request pertama setelah idle ±10–30 dtk).
- Disk container **tidak persisten** antar redeploy → setelah redeploy, jalankan lagi
  *Analisis Video → Training Model* lewat UI (prosesnya memang by-design via UI).
- Crawling 1–5 menit aman di Fly.io (bukan serverless, tidak ada timeout fungsi pendek).
