"""Generator laporan PDF bergaya makalah untuk hasil analisis sentimen.

Isi laporan:
  Cover
  BAB I   Pendahuluan (latar belakang, tujuan, batasan)
  BAB II  Metodologi / Cara Kerja (alur pipeline + tabel teknik)
  BAB III Hasil Analisis (data video, distribusi, chart, sampel komentar)
  BAB IV  Evaluasi Hasil (metrik model, confusion matrix, per kelas, interpretasi)
  BAB V   Kesimpulan dan Saran

Font bawaan reportlab (WinAnsi/Latin-1) dipakai, jadi semua teks disaring
melalui _t() agar karakter di luar Latin-1 tidak merusak PDF.
"""
import os
import re
import html
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, HRFlowable,
)

# ==================== UTIL TEKS ====================
_CHAR_MAP = {
    "—": "-", "–": "-", "≈": "~", "×": "x", "≤": "<=", "≥": ">=",
    "“": '"', "”": '"', "‘": "'", "’": "'", "…": "...", " ": " ",
    "↑": "^", "↓": "v", "✓": "v", "⚠": "[!]", "→": "->",
}


def _t(s) -> str:
    """Sanitasi teks: aman untuk font Latin-1 reportlab + escape XML."""
    if s is None:
        return ""
    s = str(s)
    for a, b in _CHAR_MAP.items():
        s = s.replace(a, b)
    # buang karakter di luar Latin-1 (mis. emoji)
    s = "".join(ch for ch in s if ord(ch) < 256)
    return html.escape(s, quote=False)


def _pct(part: int, total: int) -> str:
    return f"{part / total * 100:.1f}%" if total else "0.0%"


def _img_if(path: str, width_cm: float = 14):
    if path and os.path.exists(path):
        try:
            from PIL import Image as PILImage
            with PILImage.open(path) as im:
                w, h = im.size
        except Exception:
            w, h = 700, 400
        img = Image(path, width=width_cm * cm, height=width_cm * cm * h / w)
        return img
    return None


# ==================== GAYA ====================
def _styles():
    base = getSampleStyleSheet()
    s = {}
    s["cover_title"] = ParagraphStyle("cover_title", parent=base["Title"],
                                      fontSize=22, leading=28, alignment=TA_CENTER,
                                      textColor=colors.HexColor("#1e3a8a"), spaceAfter=10)
    s["cover_sub"] = ParagraphStyle("cover_sub", parent=base["Normal"],
                                    fontSize=13, leading=18, alignment=TA_CENTER,
                                    textColor=colors.HexColor("#334155"), spaceAfter=6)
    s["cover_author"] = ParagraphStyle("cover_author", parent=base["Normal"],
                                       fontSize=15, leading=20, alignment=TA_CENTER,
                                       fontName="Helvetica-Bold",
                                       textColor=colors.HexColor("#0f172a"))
    s["small_bold"] = ParagraphStyle("small_bold", parent=base["Normal"], fontSize=9.5,
                                     leading=13, alignment=TA_CENTER, fontName="Helvetica-Bold",
                                     textColor=colors.HexColor("#475569"))
    s["small_c"] = ParagraphStyle("small_c", parent=base["Normal"], fontSize=9.5,
                                  leading=13, alignment=TA_CENTER,
                                  textColor=colors.HexColor("#475569"))
    s["bab"] = ParagraphStyle("bab", parent=base["Heading1"], fontSize=15,
                              textColor=colors.HexColor("#1e3a8a"),
                              spaceBefore=18, spaceAfter=8)
    s["sub"] = ParagraphStyle("sub", parent=base["Heading2"], fontSize=12,
                              textColor=colors.HexColor("#334155"),
                              spaceBefore=10, spaceAfter=4)
    s["body"] = ParagraphStyle("body", parent=base["Normal"], fontSize=10,
                               leading=15, alignment=TA_JUSTIFY, spaceAfter=6)
    s["bullet"] = ParagraphStyle("bullet", parent=s["body"],
                                 leftIndent=14, bulletIndent=4, spaceAfter=3)
    s["num"] = ParagraphStyle("num", parent=s["body"],
                              leftIndent=18, bulletIndent=4, spaceAfter=4)
    s["small"] = ParagraphStyle("small", parent=base["Normal"], fontSize=8.5,
                                leading=12, textColor=colors.HexColor("#64748b"))
    s["caption"] = ParagraphStyle("caption", parent=base["Normal"], fontSize=8.5,
                                  leading=12, alignment=TA_CENTER,
                                  textColor=colors.HexColor("#475569"),
                                  spaceBefore=4, spaceAfter=10)
    s["cell"] = ParagraphStyle("cell", parent=base["Normal"], fontSize=8.5, leading=12)
    s["cellc"] = ParagraphStyle("cellc", parent=s["cell"], alignment=TA_CENTER)
    return s


ST = _styles()


def _table(data, col_widths=None, header=True):
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    t.setStyle(TableStyle(style))
    return t


# ==================== PEMBUAT LAPORAN ====================
def build_report_pdf(out_path: str, video: dict, analysis: dict,
                     metrics: dict = None, charts_dir: str = "",
                     settings: dict = None):
    """Bangun file PDF laporan. `video` dan `analysis` dari hasil analisis URL;
    `metrics` dari training model (bisa None jika model belum dilatih);
    `settings` identitas penulis (nama, nim, prodi, dosen) untuk cover."""
    settings = settings or {}
    nama = str(settings.get("nama", "") or "").strip()
    nim = str(settings.get("nim", "") or "").strip()
    prodi = str(settings.get("prodi", "") or "").strip()
    dosen = str(settings.get("dosen", "") or "").strip()
    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title="Laporan Analisis Sentimen", author=nama or "SentimenYT",
    )
    E = []  # elements

    total = int(analysis.get("total", 0) or 0)
    dist = analysis.get("dist_all", {}) or {}
    n_pos = int(dist.get("positif", 0) or 0)
    n_net = int(dist.get("netral", 0) or 0)
    n_neg = int(dist.get("negatif", 0) or 0)
    dominant = max(dist, key=dist.get) if dist else "netral"
    samples = (analysis.get("samples") or [])[:15]
    has_model = bool(analysis.get("model_available"))
    today = datetime.now().strftime("%d %B %Y")

    # ---------- COVER ----------
    E.append(Spacer(1, 3.2 * cm))
    E.append(Paragraph("LAPORAN HASIL ANALISIS SENTIMEN", ST["cover_title"]))
    E.append(HRFlowable(width="60%", thickness=1.2, color=colors.HexColor("#1e3a8a"),
                        spaceAfter=10, spaceBefore=4))
    E.append(Paragraph("Komentar YouTube terhadap Kebijakan MBG<br/>(Makan Bergizi Gratis)", ST["cover_sub"]))
    E.append(Spacer(1, 1.6 * cm))
    cover_tbl = [
        ["Video", _t(video.get("title", "-"))],
        ["Channel", _t(video.get("channel", "-"))],
        ["Sumber", _t(video.get("url", "-"))],
        ["Teks dianalisis", f"{total:,} komentar/reply"],
        ["Tanggal analisis", today],
        ["Sentimen dominan", _t(dominant).capitalize()],
    ]
    ct = Table(cover_tbl, colWidths=[4.5 * cm, 11 * cm])
    ct.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10.5),
        ("ALIGN", (0, 0), (0, -1), "RIGHT"),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#475569")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    E.append(ct)

    # ---------- IDENTITAS PENULIS (cover) ----------
    if nama or nim:
        E.append(Spacer(1, 1.4 * cm))
        E.append(Paragraph("Disusun oleh:", ST["small_bold"]))
        if nama:
            E.append(Paragraph(_t(nama), ST["cover_author"]))
        if nim:
            E.append(Paragraph("NIM: " + _t(nim), ST["small_c"]))
        if prodi:
            E.append(Paragraph(_t(prodi), ST["small_c"]))
        if dosen:
            E.append(Paragraph("Dosen Pengampu: " + _t(dosen), ST["small_c"]))
    E.append(Spacer(1, 1.2 * cm))
    E.append(Paragraph("Dibuat otomatis oleh aplikasi SentimenYT", ST["small"]))
    E.append(Paragraph("Flask + Naive Bayes + SVM + Leksikon Bahasa Indonesia", ST["small"]))
    E.append(PageBreak())

    # ---------- BAB I PENDAHULUAN ----------
    E.append(Paragraph("BAB I&nbsp;&nbsp;PENDAHULUAN", ST["bab"]))
    E.append(Paragraph("1.1 Latar Belakang", ST["sub"]))
    E.append(Paragraph(
        "Program Makan Bergizi Gratis (MBG) merupakan kebijakan pemerintah yang menjadi "
        "perhatian luas masyarakat. Pembahasan publik terhadap kebijakan ini banyak terjadi di "
        "platform media sosial, termasuk kolom komentar YouTube, yang mencerminkan berbagai "
        "persepsi: dukungan, kritik, hingga pernyataan netral. Karena volume komentar dapat "
        "sangat besar, diperlukan pendekatan komputasional untuk mengukur arah sentimen "
        "secara objektif, konsisten, dan dapat diulang.", ST["body"]))
    E.append(Paragraph("1.2 Tujuan", ST["sub"]))
    for b in [
        "Mengukur distribusi sentimen (positif, netral, negatif) pada komentar video YouTube yang dianalisis.",
        "Membangun model klasifikasi sentimen (Naive Bayes dan SVM) yang dilatih dari label leksikon.",
        "Menyajikan evaluasi kuantitatif hasil analisis beserta interpretasinya dalam bentuk laporan tertulis.",
    ]:
        E.append(Paragraph(b, ST["bullet"], bulletText="-"))
    E.append(Paragraph("1.3 Batasan", ST["sub"]))
    for b in [
        "Data berasal dari komentar dan reply video YouTube yang diambil melalui YouTube Data API v3, "
        f"dengan batas maksimum sesuai halaman yang dipilih (total teks dianalisis: {total:,}).",
        "Label acuan (ground truth) dihasilkan labeling leksikon berbasis aturan, bukan anotasi manual.",
        "Komentar dianalisis berdasarkan teks saja (tanpa konteks media atau thread lanjutan).",
    ]:
        E.append(Paragraph(b, ST["bullet"], bulletText="-"))

    # ---------- BAB II METODOLOGI ----------
    E.append(Paragraph("BAB II&nbsp;&nbsp;METODOLOGI / CARA KERJA", ST["bab"]))
    E.append(Paragraph(
        "Analisis dilakukan melalui pipeline enam tahap berikut. Urutan tahap dipertahankan "
        "konsisten antara saat pengolahan data, pelatihan model, dan prediksi komentar baru.", ST["body"]))
    steps = [
        ("Pengumpulan data (crawling)",
         "Komentar top-level dan reply diambil dari video yang ditentukan melalui YouTube Data API v3 "
         "(endpoint commentThreads dan comments), dengan pengambilan reply secara paralel untuk mempercepat proses."),
        ("Pembersihan teks (cleaning)",
         "Teks dikecilkan hurufnya (lowercase), kemudian dihapus URL, mention (@), hashtag (#), angka, dan tanda baca."),
        ("Normalisasi",
         "Kata singkatan dan alay dipetakan ke bentuk baku menggunakan kamus, mis. 'gak' menjadi 'tidak', "
         "'udh' menjadi 'sudah', 'bgt' menjadi 'banget'."),
        ("Penghapusan stopword (dengan pengecualian)",
         "Stopword bahasa Indonesia dihapus menggunakan Sastrawi, namun kata negasi (tidak, bukan, belum, "
         "jangan, tanpa, kurang) dan kata penguat (banget, sangat, sekali) dipertahankan agar makna tetap utuh."),
        ("Stemming",
         "Kata di-stem dengan Sastrawi Stemmer untuk menyeragamkan bentuk kata, mis. 'membantu' menjadi 'bantu'. "
         "Bentuk hasil stem yang dipakai sebagai fitur model, sedangkan leksikon dinilai pada bentuk normalisasi."),
        ("Labeling & klasifikasi",
         "Setiap komentar diberi skor oleh leksikon sentimen Bahasa Indonesia yang memperhitungkan: "
         "negasi (jendela 3 token, mendukung negasi ganda), bigram (mis. 'luar biasa', 'terima kasih'), "
         "dan intensifier (menggandakan bobot). Skor > 0 = positif, < 0 = negatif, 0 = netral. "
         "Model Naive Bayes dan SVM (linear) dilatih pada fitur TF-IDF 1-2 gram; evaluasi memakai "
         "accuracy, macro precision/recall/F1, dan confusion matrix."),
    ]
    for i, (judul, desc) in enumerate(steps, 1):
        E.append(Paragraph(f"<b>Tahap {i} — {judul}.</b> {desc}", ST["num"]))
    E.append(Paragraph("2.1 Ringkasan Teknik", ST["sub"]))
    E.append(_table([
        ["Tahap", "Teknik/Alat", "Keterangan"],
        ["Pengumpulan", "YouTube Data API v3", "Komentar + reply, paralel"],
        ["Pembersihan", "Regex + Python", "Hapus URL, mention, angka, tanda baca"],
        ["Normalisasi", "Kamus singkatan", "Kata alay ke bentuk baku"],
        ["Stopword", "Sastrawi", "Negasi & intensifier dipertahankan"],
        ["Stemming", "Sastrawi Stemmer", "Seragamkan bentuk kata"],
        ["Labeling", "Leksikon + aturan", "Negasi, bigram, intensifier"],
        ["Fitur", "TF-IDF 1-2 gram", "Maks. 5000 fitur"],
        ["Klasifikasi", "Multinomial NB & Linear SVM", "SVM berbobot kelas seimbang"],
        ["Evaluasi", "Sklearn metrics", "Accuracy, macro P/R/F1, confusion matrix"],
    ], col_widths=[3.2 * cm, 4.8 * cm, 7.8 * cm]))

    # ---------- BAB III HASIL ----------
    E.append(Paragraph("BAB III&nbsp;&nbsp;HASIL ANALISIS", ST["bab"]))
    E.append(Paragraph("3.1 Data Sumber Video", ST["sub"]))
    E.append(_table([
        ["Atribut", "Nilai"],
        ["Judul video", _t(video.get("title", "-"))],
        ["Channel", _t(video.get("channel", "-"))],
        ["Tayang", _t(str(video.get("published_at", "-"))[:10])],
        ["Views", f"{_fmt_num(video.get('view_count'))}"],
        ["Suka", f"{_fmt_num(video.get('like_count'))}"],
        ["Total komentar (YouTube)", f"{_fmt_num(video.get('comment_count'))}"],
    ], col_widths=[5 * cm, 10.8 * cm]))
    E.append(Spacer(1, 8))

    E.append(Paragraph("3.2 Distribusi Sentimen", ST["sub"]))
    E.append(Paragraph(
        f"Dari {total:,} teks yang dianalisis ({int(analysis.get('total_comment', 0) or 0)} komentar "
        f"top-level dan {int(analysis.get('total_reply', 0) or 0)} reply), distribusi sentimen adalah:",
        ST["body"]))
    E.append(_table([
        ["Kelas", "Jumlah", "Persentase"],
        ["Positif", f"{n_pos:,}", _pct(n_pos, total)],
        ["Netral", f"{n_net:,}", _pct(n_net, total)],
        ["Negatif", f"{n_neg:,}", _pct(n_neg, total)],
    ], col_widths=[5 * cm, 5 * cm, 5 * cm]))
    img = _img_if(os.path.join(charts_dir, "distribusi_url.png") if charts_dir else "")
    if img:
        E.append(Spacer(1, 6))
        E.append(img)
        E.append(Paragraph("Gambar 1. Distribusi sentimen komentar video yang dianalisis.", ST["caption"]))

    E.append(Paragraph("3.3 Perbandingan Komentar Top-Level vs Reply", ST["sub"]))
    img2 = _img_if(os.path.join(charts_dir, "comment_vs_reply.png") if charts_dir else "")
    if img2:
        E.append(img2)
        E.append(Paragraph("Gambar 2. Perbandingan sentimen komentar utama dan reply.", ST["caption"]))

    E.append(Paragraph("3.4 Sampel Komentar dan Hasil Klasifikasi", ST["sub"]))
    E.append(Paragraph(
        "Berikut sebagian sampel komentar beserta label leksikon dan (jika model tersedia) hasil prediksi model.",
        ST["body"]))
    rows = [["#", "Komentar", "Leksikon", "Skor"]]
    if has_model:
        rows[0] += ["NB", "SVM"]
    for i, s in enumerate(samples, 1):
        txt = _t((s.get("comment_text") or ""))
        if len(txt) > 110:
            txt = txt[:107] + "..."
        rows.append([str(i), Paragraph(txt, ST["cell"]), _t(s.get("label", "-")).capitalize(),
                     str(s.get("score", 0))])
        if has_model:
            rows[-1] += [_t(s.get("pred_nb", "-")).capitalize(), _t(s.get("pred_svm", "-")).capitalize()]
    E.append(_table(rows, col_widths=[0.8 * cm, 8.2 * cm, 2.1 * cm, 1.3 * cm, 2.05 * cm, 2.05 * cm][:len(rows[0])]))
    E.append(PageBreak())

    # ---------- BAB IV EVALUASI ----------
    E.append(Paragraph("BAB IV&nbsp;&nbsp;EVALUASI HASIL", ST["bab"]))
    if has_model and metrics:
        E.append(Paragraph("4.1 Metrik Model pada Test Set", ST["sub"]))
        mrows = [["Model", "Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"]]
        for name, label in [("naive_bayes", "Naive Bayes"), ("svm", "SVM (Linear)")]:
            m = metrics.get(name)
            if not m:
                continue
            mrows.append([label, f"{m['accuracy']*100:.1f}%",
                          f"{m['precision_macro']*100:.1f}%", f"{m['recall_macro']*100:.1f}%",
                          f"{m['f1_macro']*100:.1f}"])
        E.append(_table(mrows, col_widths=[4.2 * cm, 2.9 * cm, 3.4 * cm, 3.2 * cm, 2.8 * cm]))
        di = metrics.get("dataset_info", {})
        E.append(Paragraph(
            f"Model dilatih dari {di.get('total_samples', 0):,} sampel "
            f"(train {di.get('train_samples', 0):,}, test {di.get('test_samples', 0):,}).", ST["body"]))

        E.append(Paragraph("4.2 Confusion Matrix", ST["sub"]))
        cmatrix = metrics.get("svm", {}).get("confusion_matrix")
        labels = metrics.get("svm", {}).get("labels", [])
        if cmatrix and labels:
            E.append(_table([["Actual / Prediksi"] + [l.capitalize() for l in labels]] +
                            [[l.capitalize()] + [str(v) for v in row] for l, row in zip(labels, cmatrix)],
                            col_widths=[4.2 * cm] + [2.4 * cm] * len(labels)))
            E.append(Paragraph("Tabel 1. Confusion matrix model SVM (baris = aktual, kolom = prediksi).", ST["caption"]))
        img3 = _img_if(os.path.join(charts_dir, "cm_svm.png") if charts_dir else "")
        if img3:
            E.append(img3)
            E.append(Paragraph("Gambar 3. Visualisasi confusion matrix SVM.", ST["caption"]))

        E.append(Paragraph("4.3 Performa per Kelas (SVM)", ST["sub"]))
        prows = [["Kelas", "Precision", "Recall", "F1", "Support"]]
        for cls in ["positif", "netral", "negatif"]:
            r = metrics.get("svm", {}).get("classification_report", {}).get(cls)
            if r:
                prows.append([cls.capitalize(), f"{r['precision']:.2f}",
                              f"{r['recall']:.2f}", f"{r['f1-score']:.2f}", str(r["support"])])
        E.append(_table(prows, col_widths=[3.5 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm]))

        E.append(Paragraph("4.4 Kesesuaian Leksikon dan Model pada Video Ini", ST["sub"]))
        ag_nb = analysis.get("agreement_nb")
        ag_svm = analysis.get("agreement_svm")
        E.append(Paragraph(
            f"Pada data video ini, prediksi Naive Bayes sesuai dengan label leksikon pada "
            f"{ag_nb if ag_nb is not None else '-'}% teks, dan SVM pada "
            f"{ag_svm if ag_svm is not None else '-'}% teks. Tingkat kesesuaian yang tinggi "
            "menunjukkan model berhasil menangkap pola kata-kata sentimen yang dipelajari "
            "selama pelatihan.", ST["body"]))

        # Interpretasi otomatis
        nb_f1 = metrics.get("naive_bayes", {}).get("f1_macro")
        svm_f1 = metrics.get("svm", {}).get("f1_macro")
        better = "SVM (Linear)" if (svm_f1 or 0) >= (nb_f1 or 0) else "Naive Bayes"
        E.append(Paragraph("4.5 Interpretasi", ST["sub"]))
        E.append(Paragraph(
            f"Berdasarkan metrik di atas, model <b>{better}</b> memberikan F1 macro tertinggi "
            f"({max(svm_f1 or 0, nb_f1 or 0):.3f}) sehingga direkomendasikan sebagai model utama. "
            f"Sentimen dominan pada video ini adalah <b>{_t(dominant)}</b> "
            f"({_pct(dist.get(dominant, 0), total)} dari total teks), yang mencerminkan arah "
            "perbincangan mayoritas pengomentar terhadap topik yang dibahas video. "
            "Perlu dicatat bahwa label acuan berasal dari leksikon; oleh karena itu angka akurasi "
            "di atas mengukur kemampuan model mereproduksi keputusan leksikon, dan interpretasi "
            "sebaiknya tetap didukung pemeriksaan manual terhadap sejumlah sampel komentar.", ST["body"]))
    else:
        E.append(Paragraph(
            "Evaluasi model belum tersedia karena model Naive Bayes dan SVM belum dilatih "
            "(menu Training Model). Laporan ini menyajikan hasil analisis berbasis leksikon. "
            "Untuk melengkapi bab ini dengan metrik model, jalankan Training Model terlebih dahulu.",
            ST["body"]))

    # ---------- BAB V KESIMPULAN ----------
    E.append(Paragraph("BAB V&nbsp;&nbsp;KESIMPULAN DAN SARAN", ST["bab"]))
    E.append(Paragraph("5.1 Kesimpulan", ST["sub"]))
    for b in [
        f"Analisis {total:,} teks komentar/reply video \"{_t((video.get('title') or '')[:80])}\" "
        f"menghasilkan sentimen dominan <b>{_t(dominant)}</b>: "
        f"positif {n_pos:,} ({_pct(n_pos, total)}), netral {n_net:,} ({_pct(n_net, total)}), "
        f"negatif {n_neg:,} ({_pct(n_neg, total)}).",
        "Pipeline analisis (crawling, pembersihan, normalisasi, stopword, stemming, labeling leksikon) "
        "berjalan otomatis dan konsisten untuk data baru maupun komentar yang diuji ulang.",
    ] + ([
        f"Model {better} dipilih sebagai model utama dengan F1 macro "
        f"{max(svm_f1 or 0, nb_f1 or 0):.3f} pada test set, dan kesesuaian prediksi model terhadap "
        f"label leksikon pada video ini mencapai {ag_svm if ag_svm is not None else '-'}%.",
    ] if (has_model and metrics) else []) + [
        "Hasil analisis dapat dipantau melalui dashboard web serta diekspor sebagai laporan PDF ini.",
    ]:
        E.append(Paragraph(b, ST["bullet"], bulletText="-"))
    E.append(Paragraph("5.2 Saran", ST["sub"]))
    for b in [
        "Melakukan anotasi manual sebagian komentar sebagai label acuan independen untuk mengukur kualitas leksikon.",
        "Memperluas leksikon dan aturan negasi/intensifier agar cangkupan kata sentimen lebih luas.",
        "Mencoba model lanjutan (mis. fine-tuning model bahasa) jika ukuran data memungkinkan.",
    ]:
        E.append(Paragraph(b, ST["bullet"], bulletText="-"))

    # ---------- FOOTER ----------
    def _footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#94a3b8"))
        canvas.drawString(2 * cm, 1.1 * cm,
                          f"Laporan Analisis Sentimen - {today}")
        canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Halaman {doc_.page}")
        canvas.restoreState()

    doc.build(E, onFirstPage=_footer, onLaterPages=_footer)
    return out_path


def _fmt_num(v) -> str:
    try:
        return f"{int(float(v)):,}"
    except (TypeError, ValueError):
        return "0"
