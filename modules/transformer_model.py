"""Fine-tuning BERT untuk klasifikasi sentimen.

  * Bahasa Indonesia -> **IndoBERT** (`indobenchmark/indobert-lite-p1` secara
    default; bisa diganti `indobenchmark/indobert-base-p1` lewat .env)
  * Bahasa Inggris  -> **BERT** (`bert-base-uncased`)

Modul ini OPSIONAL. Tanpa `torch` + `transformers` aplikasi tetap jalan
(Naive Bayes / SVM / LSTM / GRU tidak terpengaruh).

Catatan jujur (penting untuk laporan):
  * Fine-tuning BERT untuk tugas besar butuh ribuan sampel. Dengan data
    ratusan komentar, BERT sering **tidak lebih baik** daripada TF-IDF+SVM,
    dan lebih mudah overfit. Hasil perbandingan inilah yang menarik untuk
    dibahas di laporan.
  * Training memakai CPU: ±10–40 detik/epoch untuk 500 komentar
    (varian *lite*), sedangkan *base* bisa 5–10x lebih lama.
  * Model diunduh sekali dari HuggingFace lalu di-cache di komputer.
"""
import json
import os
import random

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)
    TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    Dataset = object
    TRANSFORMERS_AVAILABLE = False

MIN_SAMPLES = 20          # total data minimum untuk BERT
MIN_PER_LANG = 10         # minimum per bahasa pada mode campuran
SEED = 42
MAX_LEN = 128

# Model default per bahasa (bisa dioverride lewat .env)
DEFAULT_MODELS = {
    # lite-base: 4 layer, ringan untuk CPU (nama resmi di HuggingFace)
    "id": "indobenchmark/indobert-lite-base-p1",
    "en": "distilbert-base-uncased",
}

# Pilihan yang lebih akurat/berat (isi BERT_MODEL_ID / BERT_MODEL_EN di .env)
ALT_MODELS = {
    "id": "indobenchmark/indobert-base-p1",
    "en": "bert-base-uncased",
}


class TransformerError(ValueError):
    """Error yang aman ditampilkan ke pengguna."""


def _quiet():
    """Redam log berisik dari transformers/huggingface."""
    try:
        import logging
        import os as _os
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        _os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        _os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        _os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    except Exception:
        pass


def _require():
    if not TRANSFORMERS_AVAILABLE:
        raise TransformerError(
            "PyTorch/Transformers belum terinstall, jadi BERT/IndoBERT tidak bisa "
            "dipakai. Jalankan di terminal:  pip install -r requirements-dl.txt\n"
            "Aplikasi lain (Naive Bayes, SVM, LSTM/GRU) tetap bisa dipakai."
        )


def model_dir_for(base_dir: str, lang: str) -> str:
    return os.path.join(base_dir, f"bert_{lang}")


def model_name(lang: str) -> str:
    env = os.getenv(f"BERT_MODEL_{lang.upper()}", "").strip()
    if env:
        return env
    return DEFAULT_MODELS.get("en" if str(lang).lower().startswith("en") else "id")


def is_trained(base_dir: str, lang: str) -> bool:
    d = model_dir_for(base_dir, lang)
    return os.path.exists(os.path.join(d, "model.safetensors")) or \
        os.path.exists(os.path.join(d, "pytorch_model.bin"))


# ================================ DATASET ================================
if TRANSFORMERS_AVAILABLE:
    class _TextDS(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
            self.enc = tokenizer(list(texts), truncation=True, padding=True,
                                 max_length=max_len, return_tensors="pt")
            self.y = labels

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            item = {k: v[i] for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.y[i], dtype=torch.long)
            return item
else:  # pragma: no cover
    _TextDS = None


# ================================ TRAINING ================================
def _finetune(texts, y, labels, name, epochs=3, batch_size=8, lr=2e-5,
              on_epoch=None, on_log=None):
    """Fine-tune satu model transformer. Return (model, tokenizer, labels)."""
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=len(labels), ignore_mismatched_sizes=True,
        id2label={i: lab for i, lab in enumerate(labels)},
        label2id={lab: i for i, lab in enumerate(labels)})
    # Checkpoint asli (mis. IndoBERT-lite) dilatih untuk 5 kelas sentimen,
    # jadi bobot klasifikasi lama dibuang & diganti 3 kelas milik kita.
    model.config.id2label = {i: lab for i, lab in enumerate(labels)}
    model.config.label2id = {lab: i for i, lab in enumerate(labels)}
    model.config.num_labels = len(labels)
    device = torch.device("cpu")
    model.to(device)

    ds = _TextDS(texts, list(y), tok)
    bs = max(2, min(batch_size, len(ds)))
    loader = DataLoader(ds, batch_size=bs, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = max(1, len(loader) * epochs)
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps)

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            out = model(**batch)
            out.loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += float(out.loss.detach())
            if on_log and step % 10 == 0:
                on_log(f"  langkah {step}/{len(loader)} — loss {running / step:.4f}")
        if on_epoch:
            on_epoch(ep, epochs, None)
        if on_log:
            on_log(f"  epoch {ep}/{epochs} selesai — loss {running / max(1, len(loader)):.4f}")

    model.eval()
    return model, tok, list(labels)


def train_bert(df: pd.DataFrame, text_col: str, label_col: str, base_dir: str,
               lang: str = "id", epochs: int = 3, batch_size: int = 8,
               lr: float = 2e-5, on_epoch=None, on_log=None):
    """Fine-tune BERT/IndoBERT. Return dict metrik (format sama dengan NB/SVM)."""
    _require()
    _quiet()

    if df is None or df.empty:
        raise TransformerError("Data kosong. Lakukan preprocessing terlebih dahulu.")
    if len(df) < MIN_SAMPLES:
        raise TransformerError(
            f"BERT butuh minimal {MIN_SAMPLES} data berlabel (sekarang {len(df)}). "
            f"Crawl lebih banyak komentar dulu."
        )

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support, classification_report)

    lang = "en" if str(lang).lower().startswith("en") else "id"
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    texts = df[text_col].astype(str).tolist()
    y_str = df[label_col].astype(str).tolist()
    labels = sorted(set(y_str))
    if len(labels) < 2:
        raise TransformerError("Data hanya punya satu kelas sentimen; tidak bisa dilatih.")
    label2id = {lab: i for i, lab in enumerate(labels)}
    y = [label2id[v] for v in y_str]

    counts = pd.Series(y_str).value_counts()
    stratify = y if counts.min() >= 4 else None
    tr_idx, te_idx = train_test_split(
        list(range(len(texts))), test_size=0.2, random_state=SEED, stratify=stratify)

    name = model_name(lang)
    if on_log:
        on_log(f"Mengunduh/memuat model: {name} (sekali saja, lalu di-cache)")

    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=len(labels), ignore_mismatched_sizes=True)

    tr_texts = [texts[i] for i in tr_idx]
    tr_y = [y[i] for i in tr_idx]
    te_texts = [texts[i] for i in te_idx]
    te_y = [y[i] for i in te_idx]

    tr_ds = _TextDS(tr_texts, tr_y, tok)
    te_ds = _TextDS(te_texts, te_y, tok)
    bs = max(2, min(batch_size, len(tr_ds)))
    tr_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True)
    te_loader = DataLoader(te_ds, batch_size=max(2, bs))

    device = torch.device("cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = max(1, len(tr_loader) * epochs)
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps)

    history = []
    best_state, best_f1 = None, -1.0
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(tr_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            out = model(**batch)
            out.loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += float(out.loss.detach())
            if on_log and step % 10 == 0:
                on_log(f"epoch {ep} langkah {step}/{len(tr_loader)} — loss {running/step:.4f}")

        model.eval()
        preds = []
        with torch.no_grad():
            for batch in te_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                preds.extend(logits.argmax(1).cpu().numpy().tolist())
        f1 = precision_recall_fscore_support(
            te_y, preds, average="macro", zero_division=0)[2]
        history.append({"epoch": ep, "loss": round(running / max(1, len(tr_loader)), 4),
                        "val_f1_macro": round(float(f1), 4)})
        if on_epoch:
            on_epoch(ep, epochs, float(f1))
        if on_log:
            on_log(f"epoch {ep}/{epochs} selesai — F1 macro validasi {f1:.4f}")
        if f1 > best_f1:
            best_f1 = float(f1)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    preds = []
    with torch.no_grad():
        for batch in te_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds.extend(model(**batch).logits.argmax(1).cpu().numpy().tolist())

    p, r, f1m, _ = precision_recall_fscore_support(
        te_y, preds, average="macro", zero_division=0)
    report = classification_report(
        [labels[i] for i in te_y], [labels[i] for i in preds],
        labels=labels, zero_division=0, output_dict=True)

    metrics = {
        "accuracy": round(float(accuracy_score(te_y, preds)), 4),
        "precision_macro": round(float(p), 4),
        "recall_macro": round(float(r), 4),
        "f1_macro": round(float(f1m), 4),
        "confusion_matrix": confusion_matrix(
            te_y, preds, labels=list(range(len(labels)))).tolist(),
        "labels": labels,
        "classification_report": {k: v for k, v in report.items() if k in labels},
        "training_history": history,
        "epochs_run": len(history),
        "base_model": name,
    }

    # ===== simpan model + tokenizer (format HuggingFace standar) =====
    out_dir = model_dir_for(base_dir, lang)
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "sentimen_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"labels": labels, "lang": lang, "max_len": MAX_LEN,
                   "base_model": name, "metrics": metrics}, f,
                  indent=2, ensure_ascii=False)
    return metrics


# ================================ PREDIKSI ================================
_PRED_CACHE = {}


def _load_saved(base_dir: str, lang: str):
    key = (base_dir, lang)
    if key in _PRED_CACHE:
        return _PRED_CACHE[key]
    _require()
    _quiet()
    d = model_dir_for(base_dir, lang)
    meta_path = os.path.join(d, "sentimen_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Model BERT ({lang}) belum dilatih. Jalankan Training dengan opsi BERT.")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    tok = AutoTokenizer.from_pretrained(d)
    model = AutoModelForSequenceClassification.from_pretrained(d)
    model.eval()
    obj = (model, tok, meta)
    _PRED_CACHE[key] = obj
    return obj


def predict_bert(texts, base_dir: str, lang: str = "id"):
    """Prediksi banyak teks. Return (list_label, list_probabilitas)."""
    model, tok, meta = _load_saved(base_dir, lang)
    labels = meta.get("labels", [])
    max_len = meta.get("max_len", MAX_LEN)
    enc = tok(list(texts), truncation=True, padding=True,
              max_length=max_len, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1).numpy()
    pred = probs.argmax(1)
    return [labels[i] for i in pred], probs.tolist()


def predict_one(text: str, base_dir: str, lang: str = "id"):
    labels, probs = predict_bert([text], base_dir, lang)
    return labels[0], max(probs[0]) if probs else 0.0


# ================================ MODE CAMPURAN (ID + EN) ================================
def dir_mix(base_dir: str) -> str:
    return os.path.join(base_dir, "bert_mix")


def is_trained_mix(base_dir: str) -> bool:
    return os.path.exists(os.path.join(dir_mix(base_dir), "sentimen_meta.json"))


def train_bert_mixed(df: pd.DataFrame, text_col: str, label_col: str,
                     base_dir: str, epochs: int = 3, batch_size: int = 8,
                     lr: float = 2e-5, on_epoch=None, on_log=None,
                     min_per_lang: int = MIN_PER_LANG):
    """Latih BERT untuk dataset CAMPURAN Indonesia + Inggris.

    Strategi: deteksi bahasa tiap komentar, lalu fine-tune model sesuai bahasa
    (IndoBERT untuk baris Indonesia, BERT untuk baris Inggris). Saat prediksi,
    model dipilih otomatis berdasarkan bahasa teks masuk.

    Satu bahasa dilatih jika subsetnya >= `min_per_lang`. Bila hanya satu bahasa
    yang memenuhi syarat, model bahasa itu tetap dipakai untuk semua teks
    (dengan catatan di metrik).
    """
    _require()
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support, classification_report)
    from .lang_detect import detect_series

    if df is None or df.empty:
        raise TransformerError("Data kosong. Lakukan preprocessing terlebih dahulu.")

    texts = df[text_col].astype(str).tolist()
    y_str = df[label_col].astype(str).tolist()
    labels = sorted(set(y_str))
    if len(labels) < 2:
        raise TransformerError("Data hanya punya satu kelas sentimen; tidak bisa dilatih.")
    label2id = {lab: i for i, lab in enumerate(labels)}
    y = [label2id[v] for v in y_str]

    lang_col = df["bahasa"].astype(str).tolist() if "bahasa" in df.columns \
        else detect_series(df[text_col])
    langs = ["en" if str(l).lower().startswith("en") else "id" for l in lang_col]

    counts = pd.Series(y_str).value_counts()
    stratify = y if counts.min() >= 4 else None
    idx_tr, idx_te = train_test_split(
        list(range(len(texts))), test_size=0.2, random_state=SEED, stratify=stratify)

    if on_log:
        n_id = sum(1 for l in langs if l == "id")
        on_log(f"Dataset campuran: {n_id} komentar Indonesia, {len(langs) - n_id} Inggris")

    # ---- latih per bahasa ----
    trained = {}
    for lg in ("id", "en"):
        tr = [i for i in idx_tr if langs[i] == lg]
        total_lg = sum(1 for l in langs if l == lg)
        if total_lg < min_per_lang or len(tr) < 4:
            if on_log:
                on_log(f"Lewati {lg.upper()}: hanya {total_lg} komentar "
                       f"(minimal {min_per_lang}) atau {len(tr)} data latih")
            continue
        if on_log:
            on_log(f"Fine-tuning {model_name(lg)} pada {len(tr)} komentar {lg.upper()}…")
        model, tok, lg_labels = _finetune(
            [texts[i] for i in tr], [y[i] for i in tr], labels,
            model_name(lg), epochs=epochs, batch_size=batch_size, lr=lr, on_log=on_log)
        out_dir = model_dir_for(base_dir, lg)
        os.makedirs(out_dir, exist_ok=True)
        model.save_pretrained(out_dir, safe_serialization=True)
        tok.save_pretrained(out_dir)
        with open(os.path.join(out_dir, "sentimen_meta.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"labels": labels, "lang": lg, "max_len": MAX_LEN,
                       "base_model": model_name(lg), "mixed": True}, f,
                      indent=2, ensure_ascii=False)
        trained[lg] = out_dir
        # bebaskan RAM sebelum melatih bahasa berikutnya (penting di laptop/CPU)
        del model, tok
        import gc
        gc.collect()

    if not trained:
        raise TransformerError(
            f"Tidak ada subset bahasa yang cukup untuk BERT (butuh minimal "
            f"{min_per_lang} per bahasa). Tambah data atau pakai Naive Bayes/SVM.")

    # ---- evaluasi gabungan pada test set yang sama ----
    import gc
    by_lang = {}
    for i in idx_te:
        lg = langs[i] if langs[i] in trained else sorted(trained)[0]
        by_lang.setdefault(lg, []).append(i)

    pred_map = {}
    for lg, idxs in by_lang.items():
        _PRED_CACHE.clear()
        gc.collect()
        labs, _ = predict_bert([texts[i] for i in idxs], base_dir, lg)
        for i, lab in zip(idxs, labs):
            pred_map[i] = label2id.get(lab, 0)
        _PRED_CACHE.clear()      # lepas model bahasa ini dari RAM
        gc.collect()

    preds = [pred_map[i] for i in idx_te]

    te_y = [y[i] for i in idx_te]
    p, r, f1m, _ = precision_recall_fscore_support(
        te_y, preds, average="macro", zero_division=0)
    report = classification_report(
        [labels[i] for i in te_y], [labels[i] for i in preds],
        labels=labels, zero_division=0, output_dict=True)

    metrics = {
        "accuracy": round(float(accuracy_score(te_y, preds)), 4),
        "precision_macro": round(float(p), 4),
        "recall_macro": round(float(r), 4),
        "f1_macro": round(float(f1m), 4),
        "confusion_matrix": confusion_matrix(
            te_y, preds, labels=list(range(len(labels)))).tolist(),
        "labels": labels,
        "classification_report": {k: v for k, v in report.items() if k in labels},
        "training_history": [],
        "base_model": " + ".join(model_name(l) for l in sorted(trained)),
        "languages_trained": sorted(trained),
        "mixed": True,
    }

    os.makedirs(dir_mix(base_dir), exist_ok=True)
    with open(os.path.join(dir_mix(base_dir), "sentimen_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump({"labels": labels, "langs": sorted(trained),
                   "dirs": {k: os.path.basename(v) for k, v in trained.items()},
                   "max_len": MAX_LEN, "metrics": metrics}, f,
                  indent=2, ensure_ascii=False)
    if on_log:
        on_log(f"Model campuran siap: {metrics['accuracy']:.2%} akurasi "
               f"(bahasa: {', '.join(sorted(trained))})")
    return metrics


def predict_bert_mixed(texts, base_dir: str):
    """Prediksi campuran: bahasa tiap teks dideteksi lalu model yang sesuai dipakai."""
    from .lang_detect import detect_language
    meta_path = os.path.join(dir_mix(base_dir), "sentimen_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            "Model BERT campuran belum dilatih. Jalankan Training (model BERT, "
            "bahasa Campuran).")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    available = meta.get("langs", [])
    if not available:
        raise FileNotFoundError("Model campuran tidak memuat bahasa apa pun.")

    texts = list(texts)
    detected = [detect_language(str(t)) for t in texts]
    out_labels = [None] * len(texts)
    out_probs = [0.0] * len(texts)
    for lg in set(detected):
        idxs = [i for i, d in enumerate(detected) if d == lg]
        use = lg if lg in available else available[0]
        labs, probs = predict_bert([texts[i] for i in idxs], base_dir, use)
        for i, lab, pr in zip(idxs, labs, probs):
            out_labels[i] = lab
            out_probs[i] = pr
        _PRED_CACHE.clear()      # hemat RAM: satu model per waktu
    return out_labels, out_probs, detected


def predict_one_mixed(text: str, base_dir: str):
    """Return (label, keyakinan_tertinggi, bahasa_terdeteksi)."""
    labels, probs, langs = predict_bert_mixed([text], base_dir)
    prob = probs[0]
    conf = max(prob) if isinstance(prob, (list, tuple)) else float(prob)
    return labels[0], float(conf), langs[0]


def load_metrics_mixed(base_dir: str) -> dict:
    try:
        with open(os.path.join(dir_mix(base_dir), "sentimen_meta.json"),
                  encoding="utf-8") as f:
            return json.load(f).get("metrics", {})
    except Exception:
        return {}


def load_metrics(base_dir: str, lang: str) -> dict:
    d = model_dir_for(base_dir, lang)
    try:
        with open(os.path.join(d, "sentimen_meta.json"), encoding="utf-8") as f:
            return json.load(f).get("metrics", {})
    except Exception:
        return {}
