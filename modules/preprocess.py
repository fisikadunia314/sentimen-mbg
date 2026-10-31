"""Modul preprocessing teks bahasa Indonesia (v3 — "benar-benar bersih").

Mengikuti tahapan Text Preprocessing pada materi *Text Mining & LDA*
(Pertemuan 12): **Case Folding → Cleaning → Tokenization → Normalization →
Stopword Removal → Stemming**, dengan penambahan aturan khusus teks media
sosial karena sumber datanya komentar YouTube/Telegram/Instagram.

Yang dibersihkan pada tahap Cleaning (v2 belum menangani ini):
  * entitas HTML (&amp; &quot; &nbsp; &gt;) — dulu lolos jadi token "amp", "quot"
  * tag HTML (<p>, <br/>, <div>) — dulu lolos jadi token "pkomentarp"
  * URL (http/https/www/t.co/belink.id), mention (@user), hashtag (#tag)
  * angka & campuran alfanumerik
  * emoji, simbol, dan seluruh karakter non-alfabet (dulu emoji ikut jadi token)
  * karakter berulang berlebihan ("mantapppp" -> "mantap")
  * token sisa HTML ("amp","nbsp","br","div", ...)
  * token sampah sosial media ("wkwkwk","hehe","rt","rw","hp","rp","no")
  * token terlalu pendek (1–2 huruf) kecuali negasi/intensifier/kata penting
  * stopword Sastrawi + daftar tambahan (dulu "sih","ya","doang","bahwa" lolos)

Negasi ("tidak", "bukan", "belum", "jangan", "tanpa", "kurang") dan
intensifier ("banget", "sangat", ...) SELALU dipertahankan karena dipakai
leksikon untuk labeling sentimen (termasuk negasi ganda).

Pipeline tetap menghasilkan kolom `norm_tokens` (belum di-stem) untuk leksikon
dan `tokens` (sudah di-stem) untuk pemodelan/LDA.
"""
import html
import os
import re
import string  # noqa: F401  (kompatibilitas impor lama)

from functools import lru_cache

import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from .lexicon import (  # noqa: E402
    INTENSIFIERS, POSITIVE_WORDS, NEGATIVE_WORDS,
    POSITIVE_BIGRAMS, NEGATIVE_BIGRAMS,
)

_stemmer = StemmerFactory().create_stemmer()
_stopword = StopWordRemoverFactory().create_stop_word_remover()

# ============================== 1. CASE FOLDING & CLEANING ==============================
# Tag HTML dibuang SEBELUM tanda baca, agar tidak menyisakan huruf lengket.
HTML_TAG_RE = re.compile(r"<[^>]{0,300}>")
# URL: http(s), www, domain pendek (t.co, bit.ly), dan domain + path
URL_RE = re.compile(
    r"(?:https?://\S+"
    r"|www\.\S+"
    r"|\b[\w.-]+\.(?:com|net|org|id|co|ly|me|be|tv|info|site|online|link|app|xyz|blog)"
    r"(?:/\S*)?)",
    re.I,
)
MENTION_HASHTAG_RE = re.compile(r"[@#]\w+")
NUMBER_RE = re.compile(r"\d+")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")          # emoji, simbol, tanda baca, aksara lain
REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")        # 3+ huruf sama -> 1
# Tawa khas media sosial: wkwk, wkwwkwk, hehehe, hihi, awokwok, xixixi, kwkwkw
LAUGHTER_RE = re.compile(r"^(?:[wk]{2,}|[wk]{2,}[aeiou]?[wk]*|h[ae]h[ae]+|h[iu]h[iu]+"
                         r"|x[i]x[i]+|aw[o]k\w*|kw[aeiou]?k\w*)$")

# Sisa-sisa HTML/escape yang sering muncul sebagai token utuh
HTML_ARTIFACTS = {
    "amp", "quot", "apos", "nbsp", "lt", "gt", "ldquo", "rdquo", "lsquo", "rsquo",
    "hellip", "mdash", "ndash", "bull", "copy", "reg", "trade", "deg", "laquo",
    "raquo", "frac12", "frac14", "sup2", "sup3", "euro", "pound", "yen",
    "br", "p", "div", "span", "li", "ul", "ol", "td", "tr", "th", "table",
    "h1", "h2", "h3", "h4", "h5", "h6", "href", "src", "alt", "img", "iframe",
    "strong", "em", "font", "style", "class", "css", "json", "script", "html",
}

# ============================== 2. NORMALIZATION (kamus alay) ==============================
NORMALIZATION_DICT = {
    # negasi
    "gk": "tidak", "ga": "tidak", "gak": "tidak", "tdk": "tidak", "g": "tidak",
    "engga": "tidak", "enggak": "tidak", "nggak": "tidak", "ngga": "tidak",
    "gax": "tidak", "gakk": "tidak", "tdk": "tidak", "tak": "tidak",
    "bkn": "bukan", "bukann": "bukan",
    # kata ganti / sambung / umum
    "yg": "yang", "dgn": "dengan", "utk": "untuk", "krn": "karena", "karna": "karena",
    "sdh": "sudah", "udh": "sudah", "udah": "sudah", "blm": "belum", "belom": "belum",
    "tp": "tapi", "tpi": "tapi", "tetapi": "tapi",
    "bgt": "banget", "bngt": "banget", "bngtt": "banget", "sgt": "sangat",
    "skli": "sekali", "skali": "sekali", "bgs": "bagus", "jlk": "jelek",
    "sm": "sama", "aja": "saja", "aj": "saja", "dr": "dari", "dlm": "dalam",
    "pd": "pada", "dn": "dan", "jg": "juga", "jga": "juga", "lg": "lagi",
    "lgi": "lagi", "skrg": "sekarang", "skrng": "sekarang", "skrg2": "sekarang",
    "tau": "tahu", "taunya": "tahu", "abis": "habis", "mantul": "mantap",
    "mantab": "mantap", "kerenn": "keren", "org": "orang", "mrk": "mereka", "kalo": "kalau", "klo": "kalau", "kl": "kalau",
    "jd": "jadi", "jgn": "jangan", "mkn": "makan", "byk": "banyak",
    "bnyk": "banyak", "banyk": "banyak", "sdkt": "sedikit", "smua": "semua",
    "smw": "semua", "dpt": "dapat", "dapet": "dapat", "bs": "bisa", "bsa": "bisa",
    "hrs": "harus", "hrga": "harga", "hrg": "harga", "tsb": "tersebut",
    "trs": "terus", "trus": "terus", "msh": "masih", "msih": "masih",
    "pke": "pakai", "pk": "pakai", "pake": "pakai", "gmn": "bagaimana",
    "bgmn": "bagaimana", "gimana": "bagaimana", "knp": "kenapa",
    "kenapa": "kenapa", "dgn2": "dengan", "jg2": "juga",
    "jln": "jalan", "jlan": "jalan", "ank": "anak", "anak2": "anak",
    "pmrth": "pemerintah", "pemrintah": "pemerintah", "pemerintahan": "pemerintah",
    "kebjkn": "kebijakan", "kebijakan2": "kebijakan", "prgrm": "program",
    "progrm": "program", "prgram": "program", "anggrn": "anggaran",
    "anggarn": "anggaran", "drhana": "darah", "warga2": "warga",
    "daerah2": "daerah", "murid2": "murid", "siswa2": "siswa",
    "guru2": "guru", "sekolah2": "sekolah", "anak-anak": "anak",
    # kata yang sudah baku (jaga-jaga agar tidak salah normalisasi)
    "makanan": "makanan", "makan": "makan",
}

# ============================== 3. STOPWORD TAMBAHAN ==============================
# Kata umum yang tidak informatif untuk topik/sentimen tapi lolos dari Sastrawi.
EXTRA_STOPWORDS = {
    # partikel & filler percakapan
    "sih", "deh", "dong", "nih", "loh", "lah", "kah", "pun", "kok", "kan", "ya",
    "yah", "yaa", "yha", "tuh", "gitu", "gini", "emang", "memang", "banget2",
    "doang", "cuma", "cuman", "hanya", "saja", "juga", "pula", "lagi", "masih",
    "bisa", "boleh", "mau", "ingin", "harus", "sudah", "telah", "sedang", "akan",
    "lebih", "agak", "begitu", "demikian",
    # kata rujuk / penghubung yang tidak informatif
    "itu", "ini", "tersebut", "bahwa", "adalah", "merupakan", "ialah", "yaitu",
    "yakni", "para", "sebuah", "seorang", "sebagian", "seluruh", "semua",
    "setiap", "tiap", "beberapa", "banyak", "sedikit", "hal", "sesuatu",
    "ia", "dia", "kita", "kami", "kalian", "saya", "aku", "kamu", "anda",
    "nya", "punya", "adanya", "terhadap", "mengenai", "tentang", "bagi",
    "secara", "serta", "maupun", "namun", "tetapi", "sehingga", "karena",
    "jika", "kalau", "apabila", "ketika", "saat", "waktu", "setelah", "sebelum",
    "sampai", "hingga", "sejak", "antara", "dalam", "luar", "atas", "bawah",
    "sini", "sana", "situ", "mana", "kapan", "bagaimana", "mengapa", "kenapa",
    "siapa", "apa", "apakah", "yaitu", "misalnya", "contoh", "dsb", "dll",
    # singkatan/istilah administratif yang jadi noise di komentar
    "rt", "rw", "hp", "rp", "no", "nomor", "alamat", "wa", "sms", "telp",
    "kecamatan", "kelurahan", "desa2", "kabupaten2", "provinsi2",
    # kata kerja "meta" yang terlalu umum
    "bilang", "kata", "ucap", "ujar", "sebut", "tulis", "komentar", "koment",
    "vidio", "video", "youtube", "instagram", "telegram", "medsos", "sosmed",
    "konten", "channel", "akun", "posting", "upload", "share", "subscribe",
    "like", "view", "subscriber", "follow", "follower", "tag", "link",
    "haha", "hihi", "hehe", "huhu", "wkwk", "wkwkwk", "wkwkwkwk", "awok",
    "awokwok", "ngakak", "gokil", "anjir", "anjay", "asli",
    "adalah2", "harusnya", "seharusnya", "semoga", "mudah", "mudahan",
    "mah", "atuh", "euy", "geuning", "loh2", "pokok", "pokoknya", "intinya",
    # sapaan / panggilan (tidak informatif untuk topik & sentimen)
    "pak", "bu", "bapak", "ibu", "mas", "mbak", "kak", "bang", "om", "tante",
    "gan", "agan", "sis", "bro", "min", "admin", "guys", "teman", "teman2",
    "sih2", "wkwk2", "banget3",
}

# Kata pendek yang tetap dipertahankan (2 huruf) — mis. "ai" pada materi LDA
SHORT_ALLOW = {"ai", "ui", "tv", "ac", "ai2"}

# Negasi & intensifier tidak boleh dibuang
NEGATION_WORDS = {"tidak", "bukan", "belum", "jangan", "tanpa", "kurang"}
_KEEP_WORDS = NEGATION_WORDS | INTENSIFIERS

# Kata bermuatan sentimen TIDAK di-stem (mis. "setuju" jangan jadi "tuju"),
# supaya makna & interpretasi topik tetap utuh.
PROTECTED_WORDS = set(POSITIVE_WORDS) | set(NEGATIVE_WORDS) | _KEEP_WORDS

# Kata domain publik/kebijakan yang sering dirusak Sastrawi
# ("kebijakan" -> "bijak", "pemerintah" -> "perintah", "kesehatan" -> "sehat").
DOMAIN_WORDS = {
    "kebijakan", "pemerintah", "pemerintahan", "program", "anggaran", "negara",
    "warga", "masyarakat", "rakyat", "anak", "sekolah", "siswa", "murid", "guru",
    "makanan", "makan", "gizi", "sehat", "kesehatan", "gratis", "subsidi",
    "bantuan", "layanan", "pelayanan", "publik", "daerah", "pusat", "menteri",
    "presiden", "kementerian", "dana", "pajak", "harga", "ekonomi", "pendidikan",
    "kualitas", "fasilitas", "pembangunan", "korupsi", "kebijakan2",
    "terpencil", "pelosok", "tertinggal", "merata", "pemerataan",
}

# Gabungan kata yang HARUS selamat: sentimen + negasi + intensifier + domain +
# semua kata penyusun bigram leksikon ("luar biasa" butuh "luar" & "biasa").
_BIGRAM_WORDS = {w for p in (POSITIVE_BIGRAMS | NEGATIVE_BIGRAMS) for w in p.split()}
_PROTECTED_ALL = PROTECTED_WORDS | DOMAIN_WORDS | _BIGRAM_WORDS

# Jangan pernah membuang kata bermakna lewat daftar stopword tambahan.
EXTRA_STOPWORDS -= _PROTECTED_ALL

# Kata tambahan dari pengguna (opsional): data/stopwords_custom.txt
CUSTOM_STOPWORDS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "stopwords_custom.txt")


@lru_cache(maxsize=1)
def _custom_stopwords() -> frozenset:
    try:
        with open(CUSTOM_STOPWORDS_FILE, encoding="utf-8") as f:
            return frozenset(
                ln.strip().lower() for ln in f
                if ln.strip() and not ln.strip().startswith("#")
            )
    except Exception:
        return frozenset()


@lru_cache(maxsize=300_000)
def _stem_word(word: str) -> str:
    """Stemming per kata (di-cache: kata unik hanya di-stem sekali)."""
    return _stemmer.stem(word)


@lru_cache(maxsize=300_000)
def _is_stopword(word: str) -> bool:
    """Cek stopword Sastrawi per kata (di-cache)."""
    return _stopword.remove(word).strip() == ""


# ============================== TAHAP PREPROCESSING ==============================
def clean_text(text: str) -> str:
    """Cleaning + case folding: buang URL, HTML, mention, hashtag, angka,
    emoji/simbol, dan rapikan huruf berulang."""
    if not isinstance(text, str):
        return ""
    t = html.unescape(text)                  # &amp; -> &  (bukan token "amp")
    t = HTML_TAG_RE.sub(" ", t)              # <p>, <br/> -> spasi
    t = t.lower()                            # case folding
    t = URL_RE.sub(" ", t)                   # URL / domain
    t = MENTION_HASHTAG_RE.sub(" ", t)       # @user, #hashtag
    t = NUMBER_RE.sub(" ", t)                # angka
    t = NON_ALPHA_RE.sub(" ", t)             # emoji, tanda baca, simbol
    t = REPEAT_CHAR_RE.sub(r"\1", t)          # mantapppp -> mantap
    tokens = [w for w in t.split() if w not in HTML_ARTIFACTS]
    return " ".join(tokens)


def normalize_text(text: str) -> str:
    """Normalization: seragamkan kata alay/singkatan ke bentuk baku."""
    return " ".join(NORMALIZATION_DICT.get(t, t) for t in text.split())


def _is_noise_token(tok: str) -> bool:
    """Token yang tidak layak masuk fitur."""
    if not tok.isalpha():
        return True
    if tok in HTML_ARTIFACTS or tok in EXTRA_STOPWORDS:
        return True
    if LAUGHTER_RE.match(tok):
        return True
    if tok in _custom_stopwords():
        return True
    if len(tok) == 1:
        return True
    if len(tok) == 2 and tok not in _KEEP_WORDS and tok not in SHORT_ALLOW:
        return True
    return False


def remove_stopwords(text: str) -> str:
    """Stopword removal: Sastrawi + daftar tambahan, negasi & intensifier tetap."""
    keep = []
    for tok in text.split():
        if tok in _KEEP_WORDS:
            keep.append(tok)
            continue
        if _is_noise_token(tok):
            continue
        if _is_stopword(tok):
            continue
        keep.append(tok)
    return " ".join(keep)


def _is_stem_noise(stem: str) -> bool:
    """Stem yang tidak layak dipakai sebagai fitur."""
    if not stem:
        return True
    if stem in HTML_ARTIFACTS or stem in EXTRA_STOPWORDS:
        return True
    if len(stem) == 1:
        return True
    if len(stem) == 2 and stem not in _KEEP_WORDS and stem not in SHORT_ALLOW:
        return True
    return False


def stem_text(text: str) -> str:
    """Stemming Sastrawi, dengan proteksi kata bermuatan sentimen.

    Tanpa proteksi, Sastrawi mengubah "setuju" -> "tuju" dan "terpencil" ->
    "pencil", yang merusak makna. Kata leksikon/negasi/intensifier dibiarkan.
    """
    out = []
    for w in text.split():
        if w in _PROTECTED_ALL:
            out.append(w)
            continue
        s = _stem_word(w)
        if _is_stem_noise(s):
            continue
        out.append(s)
    return " ".join(out)


def tokenize(text: str) -> list:
    """Tokenization sederhana (spasi) — dipakai untuk kolom tokens/norm_tokens."""
    return [t for t in text.split() if t]


# ============================== DUKUNGAN BAHASA INGGRIS ==============================
# Indonesia memakai Sastrawi (stemming agresif). Inggris TIDAK di-stem karena
# BERT/LSTM sudah menangani morfologi, dan stemming Inggris justru merusak
# kata bermuatan sentimen ("pleased" -> "pleas").


def clean_text_en(text: str) -> str:
    """Cleaning + case folding versi Inggris (URL, HTML, emoji, angka dibuang)."""
    if not isinstance(text, str):
        return ""
    import html as _html
    t = _html.unescape(text)
    t = HTML_TAG_RE.sub(" ", t)
    t = t.lower()
    t = URL_RE.sub(" ", t)
    t = MENTION_HASHTAG_RE.sub(" ", t)
    t = NUMBER_RE.sub(" ", t)
    t = NON_ALPHA_RE.sub(" ", t)          # emoji/simbol/tanda baca
    t = REPEAT_CHAR_RE.sub(r"\1", t)      # loooove -> love
    tokens = [w for w in t.split() if w not in HTML_ARTIFACTS]
    return " ".join(tokens)


def normalize_text_en(text: str) -> str:
    from .lexicon_en import NORMALIZATION_DICT as EN_NORM
    out = []
    for tok in text.split():
        rep = EN_NORM.get(tok)
        if rep:
            out.extend(rep.split())      # "dont" -> "do not"
        else:
            out.append(tok)
    return " ".join(out)


@lru_cache(maxsize=1)
def _en_stopwords() -> frozenset:
    from .lexicon_en import STOPWORDS, KEEP_WORDS
    return frozenset(STOPWORDS - KEEP_WORDS)


@lru_cache(maxsize=1)
def _en_protected() -> frozenset:
    from .lexicon_en import PROTECTED_WORDS
    return frozenset(PROTECTED_WORDS)


def remove_stopwords_en(text: str) -> str:
    keep, protected, stops = [], _en_protected(), _en_stopwords()
    for tok in text.split():
        if tok in protected:
            keep.append(tok)
        elif tok in stops or len(tok) == 1:
            continue
        else:
            keep.append(tok)
    return " ".join(keep)


def stem_text_en(text: str) -> str:
    """Inggris sengaja TIDAK di-stem (lihat catatan di atas)."""
    return text


def preprocess_text(text: str, lang: str = "id") -> dict:
    """Pipeline lengkap untuk SATU teks (form prediksi & analisis URL)."""
    lang = str(lang).lower()
    if lang.startswith("mix"):
        from .lang_detect import detect_language
        lang = detect_language(text)
    if lang.startswith("en"):
        clean = clean_text_en(text)
        normalized = normalize_text_en(clean)
        no_stop = remove_stopwords_en(normalized)
        stemmed = stem_text_en(no_stop)
    else:
        clean = clean_text(text)
        normalized = normalize_text(clean)
        no_stop = remove_stopwords(normalized)
        stemmed = stem_text(no_stop)
    return {
        "clean": clean,
        "normalized": normalized,
        "no_stopword": no_stop,
        "stemmed": stemmed,
        "tokens": tokenize(stemmed),
        "norm_tokens": tokenize(no_stop),
    }


# ============================== PIPELINE DATAFRAME ==============================
def preprocess_dataframe(df: pd.DataFrame, text_col: str = "comment_text",
                         lang: str = "id") -> pd.DataFrame:
    """Jalankan seluruh tahap preprocessing pada sebuah DataFrame.

    Kolom hasil:
      clean       -> setelah cleaning + case folding
      normalized  -> setelah normalisasi kata alay
      no_stopword -> setelah stopword removal (negasi dipertahankan)
      stemmed     -> setelah stemming Sastrawi
      tokens      -> list token hasil stemming (untuk modeling/LDA)
      norm_tokens -> list token sebelum stemming (untuk leksikon sentimen)
    """
    df = df.copy()
    if text_col not in df.columns:
        raise ValueError(f"Kolom '{text_col}' tidak ditemukan pada data.")

    df = df.dropna(subset=[text_col])
    df[text_col] = df[text_col].astype(str)
    df = df.drop_duplicates(subset=[text_col])

    lang = str(lang).lower()
    if lang.startswith("mix"):
        # ===== MODE CAMPURAN: deteksi bahasa per komentar =====
        from .lang_detect import detect_series
        df["bahasa"] = detect_series(df[text_col])
        rows = []
        for bahasa, group in df.groupby("bahasa", sort=False):
            rows.append(preprocess_dataframe(
                group.drop(columns=["bahasa"]), text_col=text_col, lang=bahasa))
            rows[-1]["bahasa"] = bahasa
        out = pd.concat(rows, ignore_index=True) if rows else df.iloc[0:0]
        out = out.sample(frac=1.0, random_state=42).reset_index(drop=True) \
            if len(out) else out
        return out

    df["bahasa"] = lang
    _c = clean_text_en if lang == "en" else clean_text
    _n = normalize_text_en if lang == "en" else normalize_text
    _s = remove_stopwords_en if lang == "en" else remove_stopwords
    _st = stem_text_en if lang == "en" else stem_text

    df["clean"] = df[text_col].apply(_c)
    df["normalized"] = df["clean"].apply(_n)
    df["no_stopword"] = df["normalized"].apply(_s)
    df["stemmed"] = df["no_stopword"].apply(_st)
    df["tokens"] = df["stemmed"].apply(tokenize)
    df["norm_tokens"] = df["no_stopword"].apply(tokenize)

    # Buang dokumen yang kosong setelah dibersihkan
    df = df[df["tokens"].apply(lambda t: len(t) > 0)]

    # ===== Deteksi sarkasme (dari TEKS MENTAH — emoji & tanda kutip masih ada) =====
    try:
        from .sarcasm import analyze as _sarc
        flags, cues, scores = [], [], []
        for txt in df[text_col].astype(str):
            res = _sarc(txt)
            flags.append(bool(res["sarkasme"]))
            scores.append(int(res["skor"]))
            cues.append("; ".join(res["petunjuk"][:3]))
        df["sarkasme"] = flags
        df["sarkasme_skor"] = scores
        df["sarkasme_petunjuk"] = cues
    except Exception:
        df["sarkasme"] = False
        df["sarkasme_skor"] = 0
        df["sarkasme_petunjuk"] = ""

    return df.reset_index(drop=True)


# ============================== ALAT BANTU DIAGNOSTIK ==============================
def token_frequency(df: pd.DataFrame, col: str = "tokens", top_n: int = 30) -> list:
    """Frekuensi token (untuk memeriksa kualitas hasil preprocessing di UI)."""
    from collections import Counter
    counter = Counter()
    for toks in df.get(col, []):
        if isinstance(toks, (list, tuple)):
            counter.update(toks)
        elif isinstance(toks, str):
            counter.update(toks.split())
    return counter.most_common(top_n)


def quality_report(df: pd.DataFrame, text_col: str = "comment_text") -> dict:
    """Ringkasan kualitas: berapa dokumen kosong, rata-rata token, dll."""
    n = len(df)
    toks = df["tokens"].apply(lambda t: len(t) if isinstance(t, list) else 0) \
        if "tokens" in df.columns else pd.Series(dtype=int)
    return {
        "documents": int(n),
        "avg_tokens": round(float(toks.mean()), 2) if len(toks) else 0.0,
        "max_tokens": int(toks.max()) if len(toks) else 0,
        "min_tokens": int(toks.min()) if len(toks) else 0,
        "unique_tokens": len({t for toks in df.get("tokens", []) if isinstance(toks, list)
                              for t in toks}),
    }
