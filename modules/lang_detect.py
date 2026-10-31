"""Deteksi bahasa per komentar (Indonesia vs Inggris) tanpa dependensi eksternal.

Kenapa tidak memakai `langdetect`/`fasttext`?
  * `langdetect` butuh instalasi tambahan + tidak deterministik.
  * Komentar media sosial sangat pendek ("mantap", "not bad") sehingga model
    statistik besar justru sering salah. Untuk dua bahasa, penilaian berbasis
    kata penanda + imbuhan jauh lebih akurat dan bisa dijelaskan di laporan.

Cara kerja (skor berbobot, deterministik):
  1. Kata penanda kuat  (yang, tidak, banget, ... / the, and, not, ...)  -> +3
  2. Kata fungsi umum    (saya, untuk, ... / this, with, ...)            -> +1
  3. Imbuhan khas        (me-, -nya, -lah, ber-, ter-)                   -> +2
  4. Huruf/klaster khas  (ng, ny, "q tanpa u", ... / "th", "wh", ...)    -> +1
Skor tertinggi menang; jika keduanya 0 atau seri -> dianggap Indonesia
(bahasa mayoritas dataset).
"""

import re

# ---- kata penanda kuat (bobot 3) ----
ID_STRONG = {
    "yang", "tidak", "tidak", "dengan", "untuk", "dari", "dalam", "akan",
    "sudah", "belum", "bisa", "juga", "saja", "banget", "sekali", "sangat",
    "karena", "tetapi", "tapi", "kalau", "harus", "masih", "banyak", "semua",
    "saya", "kamu", "kita", "mereka", "dia", "apa", "bagaimana", "kenapa",
    "program", "kebijakan", "pemerintah", "warga", "anak", "makanan",
    "mantap", "bagus", "jelek", "setuju", "mubazir", "dukung", "tolak",
    "terima", "kasih", "gratis", "sekolah", "guru", "siswa", "uang",
}
EN_STRONG = {
    "the", "and", "is", "are", "was", "were", "not", "you", "your", "this",
    "that", "these", "those", "with", "have", "has", "had", "will", "would",
    "should", "could", "about", "there", "their", "they", "what", "when",
    "where", "which", "because", "but", "for", "from", "very", "really",
    "good", "bad", "great", "terrible", "love", "hate", "agree", "disagree",
    "program", "policy", "government", "people", "children", "food",
    "school", "teacher", "student", "money", "helpful", "waste", "support",
}

# ---- kata fungsi umum (bobot 1) ----
ID_WEAK = {
    "di", "ke", "dan", "atau", "ini", "itu", "ada", "pada", "oleh", "agar",
    "supaya", " jika", "jika", "seperti", "lebih", "paling", "masalah",
    "orang", "hari", "tahun", "kali", "lagi", "punya", "buat", "jadi",
}
EN_WEAK = {
    "a", "an", "of", "in", "on", "at", "to", "it", "as", "be", "been",
    "do", "does", "did", "so", "if", "then", "than", "more", "most",
    "some", "any", "all", "one", "two", "my", "our", "me", "him", "her",
    "us", "them", "who", "whom", "how", "why", "now", "just", "only",
}

# ---- imbuhan / klaster khas (bobot 2 dan 1) ----
ID_AFFIX = re.compile(
    r"\b(me|men|mem|meng|ber|ter|pe|pen|pem|peng|ke)\w{3,}"
    r"|\w+(nya|lah|kah|pun|kan)\b"
)
EN_AFFIX = re.compile(r"\b\w+(ing|tion|ment|ness|able|ible|ous|ful|less)\b")
ID_BIGRAM_CHARS = re.compile(r"\b\w*(ng|ny|ngg|sy|dh)\w*\b")
EN_BIGRAM_CHARS = re.compile(r"\b\w*(th|wh|sh|ck|ght)\w*\b")

WORD_RE = re.compile(r"[a-z']+")


def _lexicon_sets():
    """Kata bermuatan sentimen per bahasa -> sinyal kuat untuk deteksi."""
    try:
        from .lexicon import POSITIVE_WORDS as IDP, NEGATIVE_WORDS as IDN
        from .lexicon_en import POSITIVE_WORDS as ENP, NEGATIVE_WORDS as ENN
        return (IDP | IDN), (ENP | ENN)
    except Exception:      # pragma: no cover
        return set(), set()


_ID_SENT, _EN_SENT = _lexicon_sets()


def _tokens(text: str):
    return WORD_RE.findall(str(text).lower())


def score_languages(text: str):
    """Return (skor_id, skor_en)."""
    toks = _tokens(text)
    if not toks:
        return 0, 0
    joined = " ".join(toks)
    sid = sen = 0
    for t in toks:
        if t in ID_STRONG:
            sid += 3
        elif t in EN_STRONG:
            sen += 3
        if t in ID_WEAK:
            sid += 1
        elif t in EN_WEAK:
            sen += 1
        # kata sentimen khas salah satu bahasa (mis. "impressive", "mubazir")
        if t in _ID_SENT and t not in _EN_SENT:
            sid += 3
        elif t in _EN_SENT and t not in _ID_SENT:
            sen += 3
    sid += 2 * len(ID_AFFIX.findall(joined))
    sen += 2 * len(EN_AFFIX.findall(joined))
    sid += len(ID_BIGRAM_CHARS.findall(joined))
    sen += len(EN_BIGRAM_CHARS.findall(joined))
    return sid, sen


def detect_language(text: str, default: str = "id") -> str:
    """Deteksi bahasa satu teks: 'id' atau 'en'."""
    sid, sen = score_languages(text)
    if sid == sen:
        return default
    return "id" if sid > sen else "en"


def detect_series(series, default: str = "id"):
    """Deteksi bahasa untuk pandas Series/kolom -> list kode bahasa."""
    return [detect_language(t, default) for t in series]


def language_distribution(series, default: str = "id"):
    """Ringkas distribusi bahasa: {'id': n, 'en': n}."""
    dist = {"id": 0, "en": 0}
    for lang in detect_series(series, default):
        dist[lang] = dist.get(lang, 0) + 1
    return dist
