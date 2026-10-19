"""Leksikon sentimen bahasa Indonesia (v2 — diperbaiki).

Perubahan penting dibanding versi lama:
1. Leksikon dipertahankan dalam bentuk BACAAN (belum di-stem) dan diterapkan
   pada token hasil normalisasi (pre-stemming), sehingga entri seperti
   "setuju", "membantu", "hentikan" benar-benar terdeteksi. (Di versi lama
   lexicon diterapkan pada token yang sudah di-stem sehingga banyak entri
   tidak pernah cocok, mis. "setuju" -> "tuju".)
2. Kata netral/kontekstual ("program", "gratis", "biasa", "makanan", dst.)
   DIHAPUS dari daftar sentimen agar tidak menimbulkan bias skor.
3. Bigram ("luar biasa", "terima kasih", "sia-sia") dihitung SATU kali.
4. Negasi: jendela 3 token ke belakang -> mendukung negasi ganda
   ("tidak tidak setuju" -> positif).
5. Intensifier ("banget", "sangat", dst.) menggandakan bobot kata sentimen.
"""

POSITIVE_WORDS = {
    "bagus", "baik", "bermanfaat", "membantu", "setuju", "dukung", "mendukung",
    "mantap", "keren", "hebat", "sukses", "berhasil", "maju", "senang",
    "bahagia", "apresiasi", "positif", "layak", "tepat", "cerdas", "peduli",
    "sejahtera", "makmur", "sehat", "bergizi", "enak", "murah", "mulia",
    "terbaik", "solusi", "puas", "terpuji", "membanggakan", "jaya", "berkah",
    "syukur", "lega", "bangga", "terimakasih", "menyenangkan",
    "mengharukan", "menginspirasi", "menghargai",
}

NEGATIVE_WORDS = {
    "buruk", "jelek", "gagal", "tolak", "menolak", "korupsi", "korup", "rugi",
    "bohong", "dusta", "tipu", "penipuan", "salah", "kacau", "parah",
    "kecewa", "marah", "benci", "sedih", "khawatir", "takut", "bahaya",
    "mahal", "mubazir", "pencitraan", "politis",
    "kampanye", "hentikan", "stop", "batal", "susah",
    "sulit", "masalah", "beban", "utang", "nepotisme", "rusak",
    "berantakan", "memalukan", "menyedihkan", "memprihatinkan",
    "mengecewakan", "menyesalkan", "merugikan", "merusak", "menipu",
    "membohongi", "menyalahkan", "menyakiti", "menodai", "menyesatkan",
    "menjelekkan", "menjatuhkan", "menghancurkan", "menghambat", "menghina",
    "mengejek", "memfitnah", "memuakkan", "mempertanyakan",
    "ragu", "dipertanyakan",
}

# Bigram dihitung satu kali (hindari double-count, mis. "luar" + "biasa").
POSITIVE_BIGRAMS = {"luar biasa", "terima kasih", "baik hati"}
NEGATIVE_BIGRAMS = {"sia sia", "tidak jelas", "tidak fair", "gagal total"}

NEGATION_WORDS = {"tidak", "bukan", "belum", "jangan", "tanpa", "kurang"}

INTENSIFIERS = {"banget", "sekali", "sangat", "amat", "paling"}

# ===== OPTIMASI: leksikon di-pre-kompilasi jadi dict bobot (1 lookup per token) =====
_WORD_WEIGHT = {}
for _w in POSITIVE_WORDS:
    _WORD_WEIGHT[_w] = 1
for _w in NEGATIVE_WORDS:
    _WORD_WEIGHT[_w] = -1

_BIGRAM_WEIGHT = {}
for _b in POSITIVE_BIGRAMS:
    _BIGRAM_WEIGHT[_b] = 1
for _b in NEGATIVE_BIGRAMS:
    _BIGRAM_WEIGHT[_b] = -1

del _w, _b


def _negated(tokens: list, i: int) -> bool:
    """Jumlah kata negasi dalam 3 token sebelum indeks i (ganjil = negatif)."""
    count = 0
    for j in range(max(0, i - 3), i):
        if tokens[j] in NEGATION_WORDS:
            count += 1
    return count % 2 == 1


def score_text(tokens: list) -> int:
    """Hitung skor sentimen dengan negasi, bigram, dan intensifier.

    Skor > 0  -> positif
    Skor < 0  -> negatif
    Skor == 0 -> netral
    """
    score = 0
    n = len(tokens)
    i = 0
    while i < n:
        # 1) Cek bigram terlebih dahulu
        if i + 1 < n:
            w_bigram = _BIGRAM_WEIGHT.get(f"{tokens[i]} {tokens[i + 1]}")
            if w_bigram is not None:
                score += -w_bigram if _negated(tokens, i) else w_bigram
                i += 2
                continue

        # 2) Cek unigram
        w = tokens[i]
        base = _WORD_WEIGHT.get(w)
        if base is not None:
            neg = _negated(tokens, i)
            amp = 1
            if not neg:
                # Intensifier bisa di depan ("paling bagus") atau di belakang
                # ("enak banget") — khas bahasa Indonesia.
                if (i > 0 and tokens[i - 1] in INTENSIFIERS) or \
                   (i + 1 < n and tokens[i + 1] in INTENSIFIERS):
                    amp = 2
            score += (base * amp) if not neg else (-base)
        i += 1
    return score


def label_from_score(score: int) -> str:
    if score > 0:
        return "positif"
    elif score < 0:
        return "negatif"
    return "netral"
