"""Modul preprocessing teks bahasa Indonesia (v2).

Pipeline: cleaning -> normalisasi -> stopword (negasi dipertahankan) -> stemming.
Selain itu dihasilkan kolom `norm_tokens`: token hasil normalisasi + stopword
(BELUM di-stem) — ini yang dipakai leksikon untuk labeling, karena entri
leksikon ditulis dalam bentuk bacaan, bukan bentuk stem.

OPTIMASI skala ribuan komentar: stemming & cek stopword dilakukan per KATA
UNIK dengan cache (lru_cache), sehingga kata yang berulang di ribuan komentar
hanya diproses sekali.
"""
import re
import string
from functools import lru_cache
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from .lexicon import INTENSIFIERS  # intensifier juga dipertahankan (agar skor 2x bekerja)

# Inisialisasi Sastrawi
_stemmer = StemmerFactory().create_stemmer()
_stopword = StopWordRemoverFactory().create_stop_word_remover()

# Kamus normalisasi kata alay/singkatan (dipakai pada level token utuh)
NORMALIZATION_DICT = {
    "gk": "tidak", "ga": "tidak", "gak": "tidak", "tdk": "tidak",
    "engga": "tidak", "enggak": "tidak", "nggak": "tidak", "ngga": "tidak",
    "yg": "yang", "dgn": "dengan", "utk": "untuk", "krn": "karena",
    "karna": "karena", "sdh": "sudah", "udh": "sudah", "blm": "belum",
    "tp": "tapi", "tpi": "tapi", "bgt": "banget", "bngt": "banget",
    "sm": "sama", "aja": "saja", "aj": "saja", "dr": "dari",
    "org": "orang", "mrk": "mereka", "kalo": "kalau", "kl": "kalau",
    "jd": "jadi", "jgn": "jangan", "mkn": "makan", "mkn": "makan",
    "anak2": "anak-anak", "anak²": "anak-anak",
    "mbg": "mbg", "makanan": "makanan",
}

# Kata negasi penting (dipertahankan saat stopword removal)
NEGATION_WORDS = {"tidak", "bukan", "belum", "jangan", "tanpa", "kurang"}

# Dipertahankan saat stopword removal: negasi + intensifier
_KEEP_WORDS = NEGATION_WORDS | INTENSIFIERS


@lru_cache(maxsize=300_000)
def _stem_word(word: str) -> str:
    """Stemming per kata (di-cache: kata unik hanya di-stem sekali)."""
    return _stemmer.stem(word)


@lru_cache(maxsize=300_000)
def _is_stopword(word: str) -> bool:
    """Cek stopword per kata (di-cache)."""
    return _stopword.remove(word).strip() == ""


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # hapus URL
    text = re.sub(r"@\w+|#\w+", " ", text)                 # hapus mention/hashtag
    text = re.sub(r"\d+", " ", text)                       # hapus angka
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(text: str) -> str:
    tokens = text.split()
    return " ".join(NORMALIZATION_DICT.get(t, t) for t in tokens)


def remove_stopwords(text: str) -> str:
    """Stopword Sastrawi, tapi pertahankan kata negasi & intensifier."""
    result = []
    for t in text.split():
        if t in _KEEP_WORDS or not _is_stopword(t):
            result.append(t)
    return " ".join(result)


def stem_text(text: str) -> str:
    return " ".join(_stem_word(w) for w in text.split())


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "comment_text") -> pd.DataFrame:
    """Pipeline preprocessing lengkap pada DataFrame.

    Kolom hasil: clean, normalized, no_stopword, stemmed,
    norm_tokens (untuk leksikon) dan tokens (untuk modeling).
    """
    df = df.copy()
    if text_col not in df.columns:
        raise ValueError(f"Kolom '{text_col}' tidak ditemukan pada data.")

    df = df.dropna(subset=[text_col])
    df = df.drop_duplicates(subset=[text_col])

    df["clean"] = df[text_col].apply(clean_text)
    df["normalized"] = df["clean"].apply(normalize_text)
    df["no_stopword"] = df["normalized"].apply(remove_stopwords)
    df["stemmed"] = df["no_stopword"].apply(stem_text)
    df["tokens"] = df["stemmed"].apply(lambda x: x.split())
    df["norm_tokens"] = df["no_stopword"].apply(lambda x: x.split())

    df = df[df["stemmed"].str.strip() != ""]
    return df.reset_index(drop=True)
