"""Pelabelan otomatis berbasis leksikon (Indonesia, Inggris, atau campuran).

Skor dihitung dari `norm_tokens` (token hasil normalisasi, belum di-stem) agar
entri leksikon dalam bentuk bacaan ("setuju", "helpful") cocok.

Penanganan sarkasme
-------------------
Komentar sindiran seperti "mantap betul, sampai rakyat makin susah" akan diberi
skor POSITIF oleh leksikon (kata "mantap"), padahal maksudnya negatif. Bila
kolom `sarkasme` bernilai True, polaritas dibalik. Skor asli tetap disimpan di
`score_leksikon`, skor akhir di `score`, sehingga bisa diaudit manual.
"""
import pandas as pd

from .lexicon import score_text, label_from_score
from .lexicon_en import score_text_en
from .lang_detect import detect_language


def score_tokens(tokens, lang: str = "id") -> int:
    if str(lang).lower().startswith("en"):
        return score_text_en(list(tokens))
    return score_text(list(tokens))


def score_tokens_mixed(tokens) -> int:
    """Skor memakai kedua leksikon; ambil yang paling informatif."""
    toks = list(tokens)
    s_id = score_text(toks)
    s_en = score_text_en(toks)
    if s_id == 0:
        return s_en
    if s_en == 0:
        return s_id
    return s_id if abs(s_id) >= abs(s_en) else s_en


def _score_row(tokens, bahasa: str, lang: str) -> int:
    if lang.startswith("mix"):
        return score_tokens_mixed(tokens)
    return score_tokens(tokens, lang)


def label_dataframe(df: pd.DataFrame, tokens_col: str = "norm_tokens",
                    lang: str = "id", handle_sarcasm: bool = True) -> pd.DataFrame:
    """Beri skor & label. Sarkasme (bila ada) membalik polaritas."""
    df = df.copy()
    lang = str(lang).lower()

    if lang.startswith("mix") and "bahasa" not in df.columns:
        df["bahasa"] = df.get("comment_text", "").apply(
            lambda t: detect_language(str(t)))
    if "bahasa" not in df.columns:
        df["bahasa"] = "id" if not lang.startswith("en") else "en"

    base_scores = []
    for tokens, bahasa in zip(df[tokens_col], df["bahasa"]):
        s = _score_row(tokens, str(bahasa), lang)
        # mode non-campuran: kalau skor 0, coba leksikon bahasa lain
        if s == 0 and not lang.startswith("mix"):
            s = score_tokens(tokens, "en" if lang.startswith("id") else "id")
        base_scores.append(s)
    df["score_leksikon"] = base_scores

    if "sarkasme" not in df.columns:
        df["sarkasme"] = False
    sarc = df["sarkasme"].astype(bool) if handle_sarcasm else pd.Series(
        [False] * len(df), index=df.index)

    def _final(s, is_s):
        """Sarkasme -> polaritas dibalik.

        * skor positif (pujian sindiran) -> negatif
        * skor 0 (pujian & keluhan saling meniadakan, mis. "mantap... tapi
          rakyat makin susah") -> negatif minimal -1, karena petunjuk sarkasme
          menunjukkan maksud sebenarnya negatif
        * skor negatif -> dibiarkan (sudah sesuai maksud)
        """
        if not is_s:
            return s
        if s > 0:
            return -s
        if s == 0:
            return -1
        return s

    df["score"] = [_final(s, bool(is_s))
                   for s, is_s in zip(df["score_leksikon"], sarc)]
    df["label"] = df["score"].apply(label_from_score)
    df["sarkasme_dibalik"] = [
        bool(is_s and (sc != s))
        for s, sc, is_s in zip(df["score_leksikon"], df["score"], sarc)
    ]
    return df
