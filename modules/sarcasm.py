"""Deteksi sarkasme / ironi berbasis petunjuk (rule-based) — ID & EN.

Sarkasme = menyatakan hal positif padahal maksudnya negatif (atau sebaliknya).
Model leksikon biasa akan salah memberi label, mis.:

    "mantap betul program ini, sampai rakyat makin susah"   -> leksikon: positif
    "hebat ya, katanya gratis tapi tagihan tetap datang"    -> leksikon: positif
    "terima kasih pak, sekarang malah makin mahal"          -> leksikon: positif

Pendekatan (jujur: ini heuristik, bukan model):
  1. **Petunjuk eksplisit** (bobot): "hebat ya", "mantap betul", "katanya",
     "padahal", "sampai" + kata negatif, emoji ironi (🙃 😏 👏 🤡), "!!" berlebih,
     elipsis "...", tanda kutip mengapit pujian, tagar #sarkas.
  2. **Konflik polaritas**: ada kata positif kuat DAN kata negatif kuat dalam
     satu komentar, disambung penanda kontras ("tapi", "padahal", "sayangnya",
     "kenyataannya", "namun") -> cenderung sarkasme.
  3. **Pujian + keluhan konkret**: kata pujian diikuti kata penderitaan
     ("makin susah", "makin mahal", "gagal", "mubazir", "korupsi").

Bila terdeteksi sarkasme dan polaritas leksikon positif -> label dibalik menjadi
negatif (dan sebaliknya). Kolom `sarkasme` disimpan agar bisa diperiksa manual.
"""
import re

# ---------------------------- PETUNJUK (ID) ----------------------------
ID_CUES = {
    # pujian yang khas dipakai menyindir
    "hebat ya": 2, "hebat banget ya": 2, "mantap betul": 2, "mantap jiwa": 2,
    "bagus banget ya": 2, "keren banget ya": 2, "pinter banget": 2,
    "pintar sekali": 1, "sukses terus": 1, "lanjutkan pak": 1,
    "terima kasih pak": 1, "terima kasih bu": 1, "terimakasih pak": 1,
    "trimakasih pak": 1, "mantap pak": 1, "hebat pak": 1, "ok mantap": 1,
    # penanda klaim / laporan orang lain (sering dipakai menyindir)
    "katanya": 2, "katanya sih": 3, "konon": 2, "kabarnya": 2,
    "dibilang": 2, "dikira": 2, "mau bagus": 2, "biar kelihatan bagus": 3,
    # penanda kontras yang menandakan keluhan setelah pujian
    "padahal": 2, "padahal mah": 3, "kenyataannya": 2, "nyatanya": 2,
    "tapi sayang": 2, "sayangnya": 2, "e tapi": 2, "tapi kenyataannya": 3,
    # penanda eksplisit
    "sarkas": 4, "sarkasme": 4, "#sarkas": 4, "garing": 2,
    "ya kan": 1, "kan katanya": 2, "gitu ya": 1, "sip lah": 1,
    # penderitaan konkret (penguat konflik)
    "makin susah": 2, "makin mahal": 2, "makin parah": 2, "makin menderita": 2,
    "rakyat kecil": 1, "uang rakyat": 1,
}

EN_CUES = {
    "yeah right": 3, "oh great": 2, "great job": 0, "well done": 0,
    "thanks a lot": 1, "sure thing": 1, "as if": 2, "supposedly": 2,
    "so called": 3, "so-called": 3, "apparently": 1, "they said": 2,
    "i guess": 1, "what a surprise": 3, "shocking": 1, "who would have thought": 3,
    "obviously": 1, "clearly": 1, "of course": 1, "/s": 4, "sarcasm": 4,
    "ironic": 3, "ironically": 3, "brilliant": 1, "genius": 1,
    "more expensive": 1, "even worse": 1, "even harder": 1,
}

# emoji/karakter ironi (dicek pada teks MENTAH, sebelum cleaning)
IRONY_CHARS = {"🙃": 3, "😏": 2, "🤡": 3, "👏": 1, "🙄": 2, "😒": 2,
               "💩": 2, "🤦": 2, "🤣": 1, "😂": 1}

# Akibat menyedihkan yang sering mengikuti pujian sindiran ("bagus... sampai antri 3 jam")
ID_CONSEQUENCE = {"antri", "mengantri", "ngantri", "nunggu", "menunggu", "telat",
                  "terlambat", "molor", "ribet", "berbelit", "susah", "mahal",
                  "menderita", "kelamaan", "berjam-jam", "sejam", "dua jam",
                  "tiga jam", "hancur", "kacau", "bobrok", "lapar", "gagal"}
EN_CONSEQUENCE = {"wait", "waiting", "queue", "queued", "hours", "late", "delay",
                  "delayed", "expensive", "suffer", "suffering", "broke",
                  "broken", "starve", "starving", "fail", "failed"}

CONTRAST_MARKERS = {"tapi", "tetapi", "padahal", "namun", "sayangnya", "but",
                    "however", "although", "though", "yet"}

QUOTED_PRAISE = re.compile(r'["\u201c\u2018\']\s*(bagus|hebat|mantap|keren|berhasil|'
                           r'sukses|good|great|success)\s*["\u201d\u2019\']', re.I)


def _norm_raw(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def _polarity_words():
    """Kata bermuatan sentimen dari kedua leksikon (untuk deteksi konflik)."""
    try:
        from .lexicon import POSITIVE_WORDS as IP, NEGATIVE_WORDS as IN
        from .lexicon_en import POSITIVE_WORDS as EP, NEGATIVE_WORDS as EN
        return (IP | EP), (IN | EN)
    except Exception:  # pragma: no cover
        return set(), set()


_POS, _NEG = _polarity_words()


def analyze(text: str, lang: str = "auto") -> dict:
    """Analisis satu teks.

    Return {"sarkasme": bool, "skor": int, "petunjuk": [..], "bahasa": "id"/"en"}
    """
    raw = str(text or "")
    low = _norm_raw(raw)
    cues_found, score = [], 0

    if not low:
        return {"sarkasme": False, "skor": 0, "petunjuk": [], "bahasa": "id"}

    # 1) petunjuk frasa (ID & EN sekaligus — aman, hanya kata yang cocok)
    for table, tag in ((ID_CUES, "id"), (EN_CUES, "en")):
        for phrase, weight in table.items():
            if phrase and phrase in low:
                if weight > 0:
                    score += weight
                    cues_found.append(f"{phrase} (+{weight})")

    # 2) emoji ironi
    for ch, weight in IRONY_CHARS.items():
        if ch in raw:
            score += weight
            cues_found.append(f"emoji {ch} (+{weight})")

    # 3) tanda baca berlebih / elipsis / pujian dalam tanda kutip
    if re.search(r"[!?]{2,}", raw):
        score += 1
        cues_found.append("tanda baca berlebih (+1)")
    if "..." in raw or "…" in raw:
        score += 1
        cues_found.append("elipsis (+1)")
    if QUOTED_PRAISE.search(raw):
        score += 2
        cues_found.append("pujian dalam tanda kutip (+2)")

    # 4) konflik polaritas dengan penanda kontras
    words = re.findall(r"[a-z']+", low)
    has_pos = any(w in _POS for w in words)
    has_neg = any(w in _NEG for w in words)
    has_contrast = any(w in CONTRAST_MARKERS for w in words)
    if has_pos and has_neg and has_contrast:
        score += 2
        cues_found.append("positif + negatif + penanda kontras (+2)")

    # 4b) "sampai ..." + akibat menyedihkan  (pola sindiran paling khas)
    if "sampai" in words or "sampe" in words or "until" in words or "till" in words:
        cons = {w for w in words if w in ID_CONSEQUENCE or w in EN_CONSEQUENCE}
        if cons or any(p in low for p in ("jam", "jam.", "jam,", "hours")):
            score += 2
            cues_found.append("pola 'sampai' + akibat menyedihkan (+2)")

    # 5) pujian di awal, keluhan panjang sesudahnya
    if has_pos and has_neg:
        pos_idx = min((i for i, w in enumerate(words) if w in _POS), default=-1)
        neg_idx = min((i for i, w in enumerate(words) if w in _NEG), default=-1)
        if 0 <= pos_idx < neg_idx and (neg_idx - pos_idx) <= 6:
            score += 1
            cues_found.append("pujian lalu keluhan berdekatan (+1)")

    return {
        "sarkasme": score >= 2,
        "skor": int(score),
        "petunjuk": cues_found[:6],
        "bahasa": "en" if score and any(1 for t in EN_CUES if t in low) and \
            not any(t in low for t in ID_CUES) else "id",
    }


def is_sarcastic(text: str) -> bool:
    return analyze(text)["sarkasme"]


def apply_polarity_flip(score: int, sarcastic: bool) -> int:
    """Sarkasme membalik polaritas: pujian sindiran -> negatif, dan sebaliknya."""
    if not sarcastic or score == 0:
        return score
    return -score


def summarize(texts) -> dict:
    """Ringkasan untuk UI: berapa komentar sarkasme & contohnya."""
    total, hits = 0, []
    for t in texts:
        total += 1
        res = analyze(t)
        if res["sarkasme"]:
            hits.append({"text": str(t)[:160], "skor": res["skor"],
                         "petunjuk": res["petunjuk"]})
    return {"total": total, "sarkasme": len(hits), "contoh": hits[:10]}
