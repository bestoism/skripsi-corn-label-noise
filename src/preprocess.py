import os
import re
import urllib.request
import pandas as pd
from src import config

# ==========================================================
# KAMUS EMOJI -> SENTIMEN
# ==========================================================
EMOJI_SENTIMENT = {
    "😭": " sedih ", "😢": " sedih ", "😔": " kecewa ",
    "😡": " marah ", "🤬": " marah ", "😤": " kesal ",
    "😍": " suka ", "❤️": " suka ", "👍": " bagus ",
    "😊": " senang ", "🙏": " terima_kasih ", "👎": " buruk ",
}

# ==========================================================
# KAMUS SLANG — DIUNDUH OTOMATIS, DI-CACHE LOKAL
# ==========================================================
KAMUS_ALAY_URL = (
    "https://raw.githubusercontent.com/nasalsabila/kamus-alay/"
    "master/colloquial-indonesian-lexicon.csv"
)
LEXICON_DIR = os.path.join(config.DRIVE_ROOT, "lexicon")
SLANG_BASE_PATH = os.path.join(LEXICON_DIR, "slang_base.csv")
SLANG_DOMAIN_PATH = os.path.join(LEXICON_DIR, "slang_domain.csv")


def _download_kamus_alay():
    """
    Unduh Kamus Alay (Salsabila dkk., 2018) sekali, simpan cache lokal di
    Drive supaya run berikutnya tidak perlu akses internet lagi.

    Rujukan efektivitas untuk IndoBERT (dikutip di Bab 3):
    - Bustamin dkk. (2025), ICIC Express Letters Part B, 16(2): normalisasi
      slang + Levenshtein sebelum IndoBERT menaikkan akurasi 3.47%.
    - Studi ablasi preprocessing IndoBERT (JCTA, 2025/2026): normalisasi
      slang adalah langkah preprocessing tunggal paling berpengaruh
      (macro F1 +0.0609 dibanding tanpa preprocessing).
    """
    os.makedirs(LEXICON_DIR, exist_ok=True)

    if os.path.exists(SLANG_BASE_PATH):
        return  # sudah ada cache, tidak perlu unduh ulang

    print(f"⬇️  Mengunduh Kamus Alay (Salsabila dkk., 2018) dari {KAMUS_ALAY_URL} ...")
    try:
        urllib.request.urlretrieve(KAMUS_ALAY_URL, SLANG_BASE_PATH)
        print(f"✅ Tersimpan di cache: {SLANG_BASE_PATH}")
    except Exception as e:
        print(f"⚠️ Gagal mengunduh Kamus Alay: {e}")
        print("   Normalisasi slang dasar akan dilewati (hanya pakai kamus domain jika ada).")


def _load_slang_lexicon():
    """
    Memuat & menggabungkan dua sumber kamus slang:
    1. slang_base.csv   -- Kamus Alay (diunduh otomatis, ~3592 entri, tidak
                            perlu kamu siapkan manual)
    2. slang_domain.csv -- opsional, entri khusus domain ulasan aplikasi yang
                            TIDAK tercakup Kamus Alay (mis. 'tokped'->
                            'tokopedia', 'cs'->'customer service'). Kalau file
                            ini tidak ada, dilewati saja, tidak error.
    """
    _download_kamus_alay()
    lexicon = {}

    if os.path.exists(SLANG_BASE_PATH):
        # FIX: file aslinya punya 7 kolom (slang,formal,In-dictionary,context,
        # category1,category2,category3) DENGAN header. usecols otomatis pakai
        # header bawaan file dan ambil kolom yang benar berdasarkan NAMA,
        # bukan posisi -- aman walau urutan/jumlah kolom sumber berubah.
        df_base = pd.read_csv(SLANG_BASE_PATH, usecols=["slang", "formal"])
        lexicon.update(dict(zip(df_base["slang"], df_base["formal"])))
        print(f"📖 Kamus slang dasar: {len(df_base)} entri (Salsabila dkk., 2018)")
    else:
        print("⚠️ Kamus slang dasar tidak tersedia -- normalisasi slang dilewati.")

    if os.path.exists(SLANG_DOMAIN_PATH):
        df_domain = pd.read_csv(SLANG_DOMAIN_PATH)  # kolom: slang, formal (dengan header)
        n_before = len(lexicon)
        lexicon.update(dict(zip(df_domain["slang"], df_domain["formal"])))
        print(f"📖 Kamus slang domain: {len(df_domain)} entri "
              f"({len(lexicon) - n_before} baru, sisanya override kamus dasar)")

    return lexicon


SLANG_DICT = _load_slang_lexicon()


# ==========================================================
# FUNGSI CLEANING
# ==========================================================
def replace_emoji_sentiment(text):
    for emo, word in EMOJI_SENTIMENT.items():
        text = text.replace(emo, word)
    return text


def clean_text_for_bert(text):
    """
    Preprocessing minimal untuk model berbasis BERT (lihat rujukan Wilie dkk.
    2020 IndoNLU; Koto dkk. 2020 IndoLEM/IndoBERT): TANPA stemming/stopword
    removal. Normalisasi slang tetap dilakukan karena divalidasi berulang
    kali sebagai langkah preprocessing paling berpengaruh untuk fine-tuning
    IndoBERT pada teks informal (Bustamin dkk., 2025; studi ablasi JCTA
    2025/2026).
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()  # wajib -- indobert-base-p1 adalah model uncased
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = replace_emoji_sentiment(text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'([!?.,])\1+', r'\1', text)

    if SLANG_DICT:
        words = text.split()
        text = ' '.join(SLANG_DICT.get(w, w) for w in words)

    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ==========================================================
# FUNGSI ANALITIK & PIPELINE UTAMA
# ==========================================================
def compute_slang_coverage(df, raw_col="review_text"):
    all_words = ' '.join(df[raw_col].dropna().astype(str).str.lower()).split()
    total = len(all_words)
    cleaned_words = [re.sub(r'[^a-z0-9]', '', w) for w in all_words]
    matched = sum(1 for w in cleaned_words if w in SLANG_DICT)
    coverage = {
        "total_words": total,
        "normalized_words": matched,
        "coverage_pct": round(matched / total * 100, 2) if total > 0 else 0.0,
    }
    print(f"📊 Cakupan kamus slang: {matched}/{total} kata ({coverage['coverage_pct']}%)")
    return coverage


def compute_text_rating_conflicts(df, text_col="cleaned_text"):
    dup_mask = df.duplicated(subset=[text_col], keep=False)
    conflicting = df[dup_mask].groupby(text_col)["rating"].nunique()
    n_conflicting = int((conflicting > 1).sum())
    print(f"🔍 Teks identik, rating berbeda: {n_conflicting} kasus")
    return n_conflicting


def drop_text_rating_conflicts(df, text_col="cleaned_text"):
    """
    Buang SEMUA baris yang teksnya identik tapi rating-nya berbeda --
    ini bukan kandidat noise (yang masih bisa didebat benar/salah),
    tapi input identik dengan 2 label kontradiktif -- secara matematis
    unlearnable untuk model apa pun. Kebijakan: buang semua baris
    yang terlibat, bukan pilih salah satu (tidak ada dasar untuk
    menentukan mana yang benar). Dokumentasikan di Batasan Penelitian.
    """
    dup_mask = df.duplicated(subset=[text_col], keep=False)
    conflict_counts = df[dup_mask].groupby(text_col)["rating"].nunique()
    conflicting_texts = conflict_counts[conflict_counts > 1].index

    n_before = len(df)
    df = df[~df[text_col].isin(conflicting_texts)].copy()
    print(f"🗑️  Baris dibuang (teks identik, rating konflik): {n_before - len(df)}")
    return df


def run_preprocessing(input_path, output_path):
    print(f"📥 Membaca data mentah dari: {input_path}")
    df = pd.read_csv(input_path)
    initial_len = len(df)
    print(f"📏 Jumlah data awal: {initial_len} baris")

    df = df.dropna(subset=["review_text"])
    print("🧹 Cleaning teks (lowercase, URL/tag, emoji, elongasi, slang)...")
    df["cleaned_text"] = df["review_text"].apply(clean_text_for_bert)

    slang_coverage = compute_slang_coverage(df)
    df = df[df["cleaned_text"].str.strip() != ""]
    
    n_conflicting = compute_text_rating_conflicts(df)
    df = drop_text_rating_conflicts(df)          # <-- BARU, sebelum drop_duplicates
    df = df.drop_duplicates(subset=["cleaned_text", "rating"])

    final_len = len(df)
    print(f"✅ Setelah dibersihkan: {final_len} baris (terbuang: {initial_len - final_len})")

    df.to_csv(output_path, index=False)
    print(f"💾 Disimpan di: {output_path}")

    summary = {
        "initial_rows": initial_len, "final_rows": final_len,
        "rows_dropped": initial_len - final_len,
        "text_rating_conflicts": n_conflicting,
        **{f"slang_{k}": v for k, v in slang_coverage.items()},
    }
    
    summary_path = os.path.join(config.RESULTS_DIR, f"preprocessing_summary__{config.DATA_VERSION}.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"💾 Ringkasan preprocessing -> {summary_path}")
    
    return df