"""
preprocess.py -- Praproses teks ulasan untuk fine-tuning IndoBERT.
Preprocessing dijaga minimal (tanpa stemming/stopword removal) sesuai
rekomendasi Wilie dkk. (2020, IndoNLU) dan Koto dkk. (2020, IndoLEM/IndoBERT)
untuk model berbasis BERT.

CATATAN: filter bahasa (Indonesia vs lainnya) sudah dilakukan di tahap
scraping (scrape_google_play.py, kolom detected_lang) -- modul ini TIDAK
mengulang filter bahasa, hanya membawa kolom detected_lang apa adanya
untuk keperluan audit.
"""

import os
import re
import urllib.request
import pandas as pd
from src import config

# ==========================================================
# KAMUS EMOJI -> SENTIMEN
# ==========================================================
# Diperluas dari versi sebelumnya (~12 entri) -- proposal menyatakan emoji
# diterjemahkan karena "umumnya tidak tercakup vocabulary WordPiece IndoBERT",
# jadi cakupannya perlu mendekati emoji yang benar-benar sering muncul di
# ulasan app store (bukan cuma subset kecil yang kebetulan terpikirkan).
EMOJI_SENTIMENT = {
    # Sedih / kecewa
    "😭": " sedih ", "😢": " sedih ", "😔": " kecewa ", "😞": " kecewa ",
    "😟": " khawatir ", "😩": " lelah ", "😫": " lelah ", "😖": " kesal ",
    "☹️": " sedih ", "🙁": " sedih ", "💔": " kecewa ",
    # Marah / kesal
    "😡": " marah ", "🤬": " marah ", "😤": " kesal ", "😑": " kesal ",
    "😠": " marah ", "👊": " marah ", "🙄": " kesal ",
    # Suka / senang / positif
    "😍": " suka ", "❤️": " suka ", "🧡": " suka ", "💛": " suka ",
    "💚": " suka ", "💙": " suka ", "💜": " suka ", "🖤": " suka ",
    "🤍": " suka ", "🤎": " suka ", "😊": " senang ", "😁": " senang ",
    "😆": " senang ", "🥰": " suka ", "😘": " suka ", "🙌": " senang ",
    "🎉": " senang ", "🎊": " senang ", "✨": " bagus ", "🔥": " bagus ",
    "💯": " bagus ", "👏": " bagus ", "👌": " bagus ", "💪": " semangat ",
    "🤝": " terima_kasih ", "😂": " lucu ", "🤣": " lucu ",
    # Terima kasih / hormat
    "🙏": " terima_kasih ",
    # Baik / buruk (evaluatif langsung)
    "👍": " bagus ", "👎": " buruk ",
    # Terkejut
    "😱": " kaget ", "😨": " takut ", "😰": " cemas ",
    # Bingung / ragu
    "😅": " canggung ", "😉": " bercanda ", "😜": " bercanda ",
    "🤔": " bingung ", "😐": " biasa_saja ",
}

# Setelah translasi, emoji yang TIDAK ada di kamus di atas (bukan berarti
# jarang -- kamus manapun selalu punya batas) di-STRIP (dihapus), bukan
# dibiarkan lolos mentah ke tokenizer. Emoji mentah yang lolos berisiko
# jadi token [UNK] atau residu aneh di WordPiece IndoBERT -- lebih aman
# dihapus (netral) daripada dibiarkan mengotori teks tanpa makna yang jelas.
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # simbol & piktograf (termasuk emoticon, transport, dll.)
    "\U00002600-\U000027BF"  # simbol lain (mis. ☹️, ✨, ❤️ sebelum variation selector)
    "\U0001F1E6-\U0001F1FF"  # bendera (regional indicator symbols)
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U00002190-\U000021FF"  # panah (jarang tapi jaga-jaga)
    "\U0000FE0F"             # variation selector (ekor emoji, mis. di ❤️)
    "]+",
    flags=re.UNICODE,
)


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

REQUIRED_SLANG_COLUMNS = ["slang", "formal"]


def _download_kamus_alay():
    """
    Unduh Kamus Alay (Salsabila dkk., 2018) sekali, simpan cache lokal di
    Drive supaya run berikutnya tidak perlu akses internet lagi.
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


def _validate_slang_columns(df, source_path):
    """
    Validasi eksplisit setelah membaca file kamus -- kalau URL sumber
    berubah format (Kamus Alay punya beberapa versi dengan jumlah/nama
    kolom berbeda), pd.read_csv(usecols=...) akan KeyError dengan pesan
    generik yang tidak menunjukkan ke mana harus dicek. Validasi ini
    menggantinya dengan pesan yang eksplisit menunjuk file dan kolom yang
    bermasalah, supaya tidak perlu debug dari traceback pandas mentah.
    """
    missing = [c for c in REQUIRED_SLANG_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Kamus slang di '{source_path}' tidak punya kolom yang diharapkan: "
            f"{missing}. Kolom yang tersedia di file ini: {df.columns.tolist()}. "
            f"Kemungkinan format sumber Kamus Alay berubah -- cek URL "
            f"({KAMUS_ALAY_URL}) secara manual dan sesuaikan REQUIRED_SLANG_COLUMNS "
            f"atau mapping kolom di _load_slang_lexicon()."
        )


def _load_slang_lexicon():
    """
    Memuat & menggabungkan dua sumber kamus slang:
    1. slang_base.csv   -- Kamus Alay (diunduh otomatis), dengan header asli
                            (slang, formal, in-dictionary, context, category1-3)
    2. slang_domain.csv -- opsional, entri khusus domain ulasan aplikasi yang
                            TIDAK tercakup Kamus Alay. Kalau file ini tidak
                            ada, dilewati saja, tidak error.
    """
    _download_kamus_alay()
    lexicon = {}

    if os.path.exists(SLANG_BASE_PATH):
        try:
            df_base = pd.read_csv(SLANG_BASE_PATH, usecols=REQUIRED_SLANG_COLUMNS)
        except ValueError as e:
            # usecols gagal cocok dengan kolom aktual -- baca tanpa usecols
            # dulu supaya bisa kasih pesan error yang jelas via _validate_slang_columns
            df_base = pd.read_csv(SLANG_BASE_PATH)
            _validate_slang_columns(df_base, SLANG_BASE_PATH)
            df_base = df_base[REQUIRED_SLANG_COLUMNS]

        lexicon.update(dict(zip(df_base["slang"], df_base["formal"])))
        print(f"📖 Kamus slang dasar: {len(df_base)} entri (Salsabila dkk., 2018)")
    else:
        print("⚠️ Kamus slang dasar tidak tersedia -- normalisasi slang dilewati.")

    if os.path.exists(SLANG_DOMAIN_PATH):
        df_domain = pd.read_csv(SLANG_DOMAIN_PATH)
        _validate_slang_columns(df_domain, SLANG_DOMAIN_PATH)
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


def strip_unrecognized_emoji(text):
    """Hapus emoji yang tidak tercakup EMOJI_SENTIMENT (dipanggil SETELAH
    replace_emoji_sentiment, supaya emoji yang sudah dikenal & diterjemahkan
    tidak ikut terhapus di sini -- hanya sisa residu yang belum tertangani)."""
    return _EMOJI_PATTERN.sub(" ", text)


_WORD_STRIP_PATTERN = re.compile(r'^(\W*)(\w+)(\W*)$', flags=re.UNICODE)


def _normalize_word_with_slang(word):
    """
    PERBAIKAN dari versi sebelumnya: text.split() sebelumnya dipanggil pada
    teks yang tanda bacanya belum dipisah dari kata (mis. "gpp," atau
    "yah!"), sehingga token seperti itu TIDAK PERNAH cocok dengan entri
    kamus ("gpp", "yah") -- cakupan normalisasi riil jauh di bawah yang
    dilaporkan compute_slang_coverage() (yang menghitung dari raw text
    dengan cara strip berbeda).

    Fix: untuk tiap token, pisahkan dulu tanda baca di PINGGIR kata (bukan
    di tengah, supaya tidak merusak kata majemuk berstrip/underscore) dari
    inti kata. Cari inti kata itu di kamus. Kalau cocok, ganti HANYA inti
    katanya, tanda baca di pinggir tetap dipertahankan apa adanya (sesuai
    prinsip preprocessing minimal -- kita tidak menghapus tanda baca,
    cukup memperbaiki pencarian kamus supaya tidak buta terhadap tanda
    baca yang menempel).
    """
    match = _WORD_STRIP_PATTERN.match(word)
    if not match:
        return word  # token tanpa karakter alfanumerik sama sekali (jarang)

    prefix, core, suffix = match.groups()
    normalized_core = SLANG_DICT.get(core, core)
    return f"{prefix}{normalized_core}{suffix}"


def clean_text_for_bert(text):
    """
    Preprocessing minimal untuk model berbasis BERT (lihat rujukan Wilie dkk.
    2020 IndoNLU; Koto dkk. 2020 IndoLEM/IndoBERT): TANPA stemming/stopword
    removal. Normalisasi slang tetap dilakukan karena divalidasi berulang
    kali sebagai langkah preprocessing paling berpengaruh untuk fine-tuning
    IndoBERT pada teks informal (Bustamin dkk., 2025).
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()  # wajib -- indobert-base-p1 adalah model uncased
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)

    text = replace_emoji_sentiment(text)
    text = strip_unrecognized_emoji(text)

    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'([!?.,])\1+', r'\1', text)

    if SLANG_DICT:
        words = text.split()
        text = ' '.join(_normalize_word_with_slang(w) for w in words)

    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ==========================================================
# FUNGSI ANALITIK & PIPELINE UTAMA
# ==========================================================
def compute_slang_coverage(df, raw_col="review_text"):
    """
    CATATAN: fungsi ini menghitung cakupan dari RAW TEXT dengan cara strip
    sendiri (re.sub kasar), BUKAN dari hasil clean_text_for_bert() -- jadi
    angka di sini adalah estimasi kasar "berapa kata yang secara teori bisa
    ternormalisasi", bukan angka final setelah pipeline lengkap. Untuk
    audit yang presisi terhadap bug pinggir-tanda-baca yang baru diperbaiki,
    bandingkan manual beberapa baris cleaned_text sebelum/sesudah fix ini.
    """
    all_words = ' '.join(df[raw_col].dropna().astype(str).str.lower()).split()
    total = len(all_words)
    cleaned_words = [re.sub(r'[^a-z0-9]', '', w) for w in all_words]
    matched = sum(1 for w in cleaned_words if w in SLANG_DICT)
    coverage = {
        "total_words": total,
        "normalized_words": matched,
        "coverage_pct": round(matched / total * 100, 2) if total > 0 else 0.0,
    }
    print(f"📊 Cakupan kamus slang (estimasi dari raw text): {matched}/{total} kata ({coverage['coverage_pct']}%)")
    return coverage


def compute_text_length_distribution(df, text_col="cleaned_text", min_words=3):
    """
    Laporan panjang teks (dalam kata) -- TIDAK memfilter otomatis (ulasan
    pendek bukan berarti label salah, cuma minim sinyal tekstual). Kolom
    'is_very_short' ditambahkan sebagai PENANDA, bukan alasan pembuangan,
    supaya nanti saat menganalisis noise di Bab IV bisa dibedakan: baris
    yang diflag CL karena teksnya memang minim informasi (bukan mismatch
    rating-teks yang sesungguhnya) vs baris yang benar-benar noise.
    """
    word_counts = df[text_col].astype(str).str.split().apply(len)
    is_very_short = word_counts <= min_words
    pct_very_short = is_very_short.mean() * 100

    print(f"📏 Distribusi panjang teks: median={word_counts.median():.0f} kata, "
          f"ulasan sangat pendek (<={min_words} kata): {pct_very_short:.2f}%")

    return word_counts, is_very_short


def compute_text_rating_conflicts(df, text_col="cleaned_text"):
    dup_mask = df.duplicated(subset=[text_col], keep=False)
    conflicting = df[dup_mask].groupby(text_col)["rating"].nunique()
    n_conflicting = int((conflicting > 1).sum())
    print(f"🔍 Teks identik, rating berbeda: {n_conflicting} kasus")
    return n_conflicting


def drop_text_rating_conflicts(df, text_col="cleaned_text"):
    """
    Untuk teks identik dengan rating berbeda: pertahankan rating MAYORITAS
    (modus) per grup teks, buang hanya baris yang menyimpang dari mayoritas
    itu. Tie-break: kalau modus tidak unik, seluruh grup dibuang (tidak ada
    dasar objektif memilih salah satu).
    """
    dup_mask = df.duplicated(subset=[text_col], keep=False)
    df_dup = df[dup_mask]
    df_unique = df[~dup_mask]

    keep_groups = []
    n_dropped_minority = 0
    n_dropped_tie = 0

    for text, group in df_dup.groupby(text_col):
        rating_counts = group["rating"].value_counts()
        if len(rating_counts) == 1:
            keep_groups.append(group)
            continue

        top_count = rating_counts.iloc[0]
        modes = rating_counts[rating_counts == top_count].index.tolist()

        if len(modes) > 1:
            n_dropped_tie += len(group)
            continue

        majority_rating = modes[0]
        kept = group[group["rating"] == majority_rating]
        n_dropped_minority += len(group) - len(kept)
        keep_groups.append(kept)

    df_result = pd.concat([df_unique] + keep_groups, ignore_index=True) if keep_groups else df_unique

    print(f"🗑️  Baris dibuang (rating minoritas dalam grup konflik): {n_dropped_minority}")
    print(f"🗑️  Baris dibuang (tie, tidak ada mayoritas jelas): {n_dropped_tie}")
    print(f"   Total dibuang: {n_dropped_minority + n_dropped_tie}")
    return df_result


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

    word_counts, is_very_short = compute_text_length_distribution(df)
    df["cleaned_text_word_count"] = word_counts
    df["is_very_short_text"] = is_very_short

    n_conflicting = compute_text_rating_conflicts(df)
    df = drop_text_rating_conflicts(df)
    df = df.drop_duplicates(subset=["cleaned_text", "rating"])

    final_len = len(df)
    print(f"✅ Setelah dibersihkan: {final_len} baris (terbuang: {initial_len - final_len})")

    df.to_csv(output_path, index=False)
    print(f"💾 Disimpan di: {output_path}")

    summary = {
        "initial_rows": initial_len, "final_rows": final_len,
        "rows_dropped": initial_len - final_len,
        "text_rating_conflicts": n_conflicting,
        "pct_very_short_text": round(is_very_short.mean() * 100, 2),
        **{f"slang_{k}": v for k, v in slang_coverage.items()},
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(config.RESULTS_DIR, "preprocessing_summary.csv"), index=False
    )
    return df