"""Pelabelan otomatis berbasis leksikon.

PENTING: skor dihitung dari `norm_tokens` (token normalisasi, belum di-stem)
agar entri leksikon dalam bentuk bacaan ("setuju", "membantu", dst.) cocok.
"""
import pandas as pd
from .lexicon import score_text, label_from_score


def label_dataframe(df: pd.DataFrame, tokens_col: str = "norm_tokens") -> pd.DataFrame:
    df = df.copy()
    df["score"] = df[tokens_col].apply(score_text)
    df["label"] = df["score"].apply(label_from_score)
    return df
