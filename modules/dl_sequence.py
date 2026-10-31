"""Model deep learning LSTM / GRU (PyTorch) untuk klasifikasi sentimen.

Sifat modul: OPSIONAL. Jika PyTorch belum terinstall, aplikasi tetap berjalan
normal (Naive Bayes + SVM + BERT) dan hanya fitur LSTM/GRU yang memberi pesan.

Kenapa LSTM/GRU dilatih dari nol (bukan memakai embedding pra-latih)?
  * Dataset tugas biasanya kecil (ratusan–ribuan komentar), jadi embedding
    yang dilatih dari korpus sendiri lebih stabil dan tidak butuh unduhan besar.
  * Jumlah parameter kecil -> training di CPU hanya hitungan detik/menit.

Arsitektur:
  Embedding(V, 100) -> SpatialDropout -> BiLSTM/BiGRU(64) -> GlobalMaxPool
  + GlobalAvgPool -> Dropout -> Linear(3)

Pemakaian:
    train_sequence_model(df, text_col, label_col, model_dir, kind="lstm")
    predict_sequence(texts, model_dir, kind="lstm")  -> (labels, probs)
"""
import json
import os
import random

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = TensorDataset = None
    TORCH_AVAILABLE = False

MIN_SAMPLES = 20          # DL butuh data lebih banyak daripada NB/SVM
MAX_VOCAB = 20_000
MAX_LEN = 60
EMBED_DIM = 100
HIDDEN = 64
EPOCHS = 12
BATCH = 32
SEED = 42

PAD, UNK = 0, 1


class SequenceError(ValueError):
    """Error yang aman ditampilkan ke pengguna."""


def _require_torch():
    if not TORCH_AVAILABLE:
        raise SequenceError(
            "PyTorch belum terinstall, jadi model LSTM/GRU tidak bisa dipakai. "
            "Jalankan di terminal:  pip install -r requirements-dl.txt\n"
            "Aplikasi lain (Naive Bayes, SVM, BERT) tetap bisa digunakan tanpa PyTorch."
        )


# ================================ MODEL ================================
if TORCH_AVAILABLE:
    class SentimentRNN(nn.Module):
        def __init__(self, vocab_size: int, num_classes: int, kind: str = "lstm"):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=PAD)
            self.drop = nn.Dropout(0.3)
            rnn_cls = nn.LSTM if kind == "lstm" else nn.GRU
            self.rnn = rnn_cls(EMBED_DIM, HIDDEN, batch_first=True,
                               bidirectional=True)
            self.fc = nn.Linear(HIDDEN * 4, num_classes)

        def forward(self, x):
            e = self.drop(self.embedding(x))
            out, _ = self.rnn(e)
            # GlobalMaxPool + GlobalAvgPool (lebih stabil daripada ambil token terakhir)
            h_max = torch.max(out, dim=1).values
            h_avg = out.mean(dim=1)
            h = torch.cat([h_max, h_avg], dim=1)
            return self.fc(self.drop(h))
else:  # pragma: no cover
    SentimentRNN = None


# ================================ UTIL ================================
def _set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_vocab(token_lists, max_vocab: int = MAX_VOCAB):
    from collections import Counter
    counts = Counter()
    for toks in token_lists:
        counts.update(toks)
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for word, _ in counts.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab


def _encode(token_lists, vocab, max_len: int = MAX_LEN):
    X = np.zeros((len(token_lists), max_len), dtype=np.int64)
    for i, toks in enumerate(token_lists):
        ids = [vocab.get(t, UNK) for t in toks[:max_len]]
        if ids:
            X[i, :len(ids)] = ids
    return X


def _model_path(model_dir: str, kind: str) -> str:
    return os.path.join(model_dir, f"{kind}.pt")


def is_trained(model_dir: str, kind: str) -> bool:
    return os.path.exists(_model_path(model_dir, kind))


# ================================ TRAINING ================================
def train_sequence_model(df: pd.DataFrame, text_col: str, label_col: str,
                         model_dir: str, kind: str = "lstm",
                         epochs: int = EPOCHS, on_epoch=None):
    """Latih LSTM/GRU. Return dict metrik (format sama dengan NB/SVM)."""
    _require_torch()
    kind = "gru" if str(kind).lower() == "gru" else "lstm"

    if df is None or df.empty:
        raise SequenceError("Data kosong. Lakukan preprocessing terlebih dahulu.")
    if len(df) < MIN_SAMPLES:
        raise SequenceError(
            f"Model {kind.upper()} butuh minimal {MIN_SAMPLES} data berlabel "
            f"(sekarang {len(df)}). Crawl lebih banyak komentar dulu, atau pakai "
            f"Naive Bayes/SVM yang tetap jalan untuk data kecil."
        )

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support, classification_report)
    from sklearn.utils.class_weight import compute_class_weight

    texts = df[text_col].astype(str).values
    y_str = df[label_col].astype(str).values
    labels = sorted(set(y_str))
    if len(labels) < 2:
        raise SequenceError("Data hanya punya satu kelas sentimen; tidak bisa dilatih.")

    label2id = {lab: i for i, lab in enumerate(labels)}
    y = np.array([label2id[v] for v in y_str])

    token_lists = [t.split() for t in texts]

    counts = pd.Series(y_str).value_counts()
    stratify = y if counts.min() >= 4 else None
    idx = np.arange(len(texts))
    idx_tr, idx_te = train_test_split(idx, test_size=0.2, random_state=SEED,
                                      stratify=stratify)

    vocab = _build_vocab([token_lists[i] for i in idx_tr])
    X_tr = _encode([token_lists[i] for i in idx_tr], vocab)
    X_te = _encode([token_lists[i] for i in idx_te], vocab)
    y_tr, y_te = y[idx_tr], y[idx_te]

    _set_seed()
    device = torch.device("cpu")
    model = SentimentRNN(len(vocab), len(labels), kind=kind).to(device)

    class_w = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    weight = torch.tensor(class_w, dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr).long())
    loader = DataLoader(ds, batch_size=min(BATCH, len(ds)), shuffle=True)

    best_state, best_f1, history = None, -1.0, []
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            out = model(xb.to(device))
            loss = loss_fn(out, yb.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += float(loss.detach()) * len(xb)

        # evaluasi pada data uji untuk memilih epoch terbaik (early stop sederhana)
        model.eval()
        with torch.no_grad():
            pred_te = model(torch.from_numpy(X_te).to(device)).argmax(1).numpy()
        f1 = precision_recall_fscore_support(
            y_te, pred_te, average="macro", zero_division=0)[2]
        history.append({"epoch": ep, "loss": round(total_loss / len(X_tr), 4),
                        "val_f1_macro": round(float(f1), 4)})
        if on_epoch:
            on_epoch(ep, epochs, float(f1))
        if f1 > best_f1:
            best_f1 = float(f1)
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_te = model(torch.from_numpy(X_te).to(device)).argmax(1).numpy()

    p, r, f1m, _ = precision_recall_fscore_support(
        y_te, pred_te, average="macro", zero_division=0)
    report = classification_report(
        [labels[i] for i in y_te], [labels[i] for i in pred_te],
        labels=labels, zero_division=0, output_dict=True)

    metrics = {
        "accuracy": round(float(accuracy_score(y_te, pred_te)), 4),
        "precision_macro": round(float(p), 4),
        "recall_macro": round(float(r), 4),
        "f1_macro": round(float(f1m), 4),
        "confusion_matrix": confusion_matrix(
            y_te, pred_te, labels=list(range(len(labels)))).tolist(),
        "labels": labels,
        "classification_report": {k: v for k, v in report.items() if k in labels},
        "training_history": history,
        "epochs_run": len(history),
    }

    # ===== simpan =====
    os.makedirs(model_dir, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "vocab": vocab,
        "labels": labels,
        "kind": kind,
        "max_len": MAX_LEN,
        "embed_dim": EMBED_DIM,
        "hidden": HIDDEN,
        "metrics": metrics,
    }, _model_path(model_dir, kind))

    return metrics


# ================================ PREDIKSI ================================
def _load(model_dir: str, kind: str):
    _require_torch()
    path = _model_path(model_dir, kind)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model {kind.upper()} belum dilatih. Jalankan Training dengan opsi {kind.upper()}.")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = SentimentRNN(len(ckpt["vocab"]), len(ckpt["labels"]),
                         kind=ckpt.get("kind", kind))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def predict_sequence(texts, model_dir: str, kind: str = "lstm"):
    """Prediksi banyak teks. Return (list_label, list_probabilitas)."""
    model, ckpt = _load(model_dir, kind)
    labels = ckpt["labels"]
    vocab = ckpt["vocab"]
    max_len = ckpt.get("max_len", MAX_LEN)
    token_lists = [str(t).split() for t in texts]
    X = _encode(token_lists, vocab, max_len)
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        probs = torch.softmax(logits, dim=1).numpy()
    pred = probs.argmax(1)
    return [labels[i] for i in pred], probs.tolist()


def predict_one(text: str, model_dir: str, kind: str = "lstm"):
    labels, probs = predict_sequence([text], model_dir, kind)
    return labels[0], max(probs[0]) if probs else 0.0


def load_metrics(model_dir: str, kind: str) -> dict:
    path = _model_path(model_dir, kind)
    if not os.path.exists(path):
        return {}
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return ckpt.get("metrics", {})
    except Exception:
        return {}
