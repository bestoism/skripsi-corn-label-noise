"""
Scraper Google Play Store, stratified per rating (1-5), dengan checkpoint
resumable, retry untuk koneksi tidak stabil, filter bahasa Indonesia inline,
dan laporan kualitas data (distribusi tanggal, panjang teks, duplikat teks)
untuk mendukung Bab III/IV.

Dijalankan SEKALI di awal (bukan bagian pipeline utama) -- setelah selesai,
hasilnya (all_reviews_master.csv) dikunci dan dipakai sebagai sumber
kebenaran tunggal untuk seluruh eksperimen berikutnya.

Cara pakai (di Colab, setelah mount Drive & clone repo):
    from scripts.scrape_google_play import main
    main()
"""

import os
import pickle
import time
import random
from datetime import datetime

import pandas as pd
from google_play_scraper import reviews, Sort

import py3langid as langid

from src import config

# ==========================================================
# KONFIGURASI SCRAPING
# ==========================================================
APPS_CONFIG = [
    {"app_id": "id.co.bankbkemobile.digitalbank", "name": "SeaBank"},
    {"app_id": "com.tokopedia.tkpd", "name": "Tokopedia"},
    {"app_id": "com.gojek.app", "name": "Gojek"},
]

# Target per rating -- dinaikkan dari revisi awal (500/500 untuk rating 2/3)
# karena kelas minoritas ini akan menyusut cukup jauh sepanjang pipeline
# (praproses -> split train -> K-Fold -> pruning). Target lebih besar di sini
# adalah buffer terhadap susut itu, BUKAN upaya menyeimbangkan distribusi
# alami rating (rating 2/3 tetap secara alami lebih jarang -- itu dilaporkan
# apa adanya, bukan dipaksa sama besar dengan rating 1/4/5).
TARGET_PER_SCORE = {1: 1200, 2: 900, 3: 900, 4: 1000, 5: 1200}

BATCH_SIZE = 200
MAX_RETRIES = 5
SLEEP_RANGE = (2, 5)

# Bahasa yang diterima -- hanya Indonesia, sesuai scope judul skripsi.
# lang="id"/country="id" pada request API HANYA memengaruhi UI Play Store
# yang dipanggil, TIDAK menjamin isi ulasan berbahasa Indonesia -- makanya
# filter eksplisit ini tetap wajib meski parameter itu sudah diset.
ACCEPTED_LANG = "id"
LANGID_MIN_CONFIDENCE = None  # py3langid tidak memberi skor probabilitas
                               # yang mudah dikalibrasi untuk teks pendek,
                               # jadi kita pakai top-1 label saja (lihat
                               # catatan _detect_language di bawah)

RAW_DIR = config.DATA_RAW_DIR
STATE_DIR = os.path.join(config.DRIVE_ROOT, "scrape_state")
os.makedirs(STATE_DIR, exist_ok=True)

ERROR_LOG_FILE = os.path.join(STATE_DIR, "scrape_errors.log")


def _log_error(app_name, score, message):
    """
    Log persisten ke file (bukan cuma print) -- supaya kalau scraping jalan
    unattended lama di Colab, kegagalan tidak baru ketahuan di akhir setelah
    semua selesai. Setiap baris: timestamp | app | rating | pesan.
    """
    with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {app_name} | "
                f"rating={score} | {message}\n")


# ==========================================================
# FILTER BAHASA -- inline, sebelum baris dihitung masuk kuota
# ==========================================================
def _detect_language(text):
    """
    py3langid dipilih dibanding langdetect karena lebih stabil untuk teks
    pendek dan informal (ulasan app store banyak yang cuma beberapa kata) --
    langdetect (berbasis Naive Bayes + iterasi acak) cenderung tidak stabil
    pada input pendek, py3langid pakai model linear pre-trained yang
    deterministik untuk input yang sama.

    KETERBATASAN YANG PERLU DICATAT DI BAB III/IV: ulasan campur kode
    (code-mixing Indonesia-Inggris berat, mis. "not bad sih tapi loadingnya
    lama banget") bisa salah terdeteksi sebagai 'en' meski secara substansi
    ulasan itu berbahasa Indonesia. Filter ini pendekatan pragmatis, bukan
    sempurna -- itu sebabnya persentase yang terbuang harus dilaporkan
    (bukan disembunyikan), supaya pembaca/penguji bisa menilai sendiri
    apakah skalanya wajar.
    """
    text = (text or "").strip()
    if len(text) < 3:
        return None  # terlalu pendek untuk dideteksi andal, lihat pemanggil
    try:
        lang, _score = langid.classify(text)
        return lang
    except Exception:
        return None


# ==========================================================
# RETRY HELPER
# ==========================================================
def _safe_scrape_call(fetch_fn, app_name, score, max_retries=MAX_RETRIES):
    """Retry untuk error koneksi (timeout, socket, connection reset ke Google)."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return fetch_fn()
        except Exception as e:
            last_error = e
            err = str(e).lower()
            transient = any(k in err for k in [
                "10035", "blockingioerror", "connectionterminated",
                "readtimeout", "timeout", "connection reset",
            ])
            wait = attempt + 2
            if transient:
                print(f"      ⏳ Kendala koneksi ({e}). Menunggu {wait}s ...")
            else:
                print(f"      ⚠️ Request gagal ({e}). Retry {attempt + 1}/{max_retries} ...")
                _log_error(app_name, score, f"Request gagal (non-transient): {e}")
            time.sleep(wait)
    _log_error(app_name, score, f"GAGAL TOTAL setelah {max_retries}x percobaan: {last_error}")
    raise RuntimeError(f"Gagal setelah {max_retries}x percobaan: {last_error}")


# ==========================================================
# CHECKPOINT — RESUMABLE (pickle, karena continuation_token adalah
# object custom _ContinuationToken, bukan tipe primitif yang bisa
# di-JSON-kan langsung)
# ==========================================================
def _state_path(app_name, score):
    return os.path.join(STATE_DIR, f"{app_name}_score{score}.pkl")


def _load_state(app_name, score):
    state_path = _state_path(app_name, score)
    checkpoint_csv = os.path.join(RAW_DIR, "_checkpoints", f"{app_name}_score{score}.csv")

    if os.path.exists(state_path) and os.path.exists(checkpoint_csv):
        with open(state_path, "rb") as f:
            state = pickle.load(f)
        df_existing = pd.read_csv(checkpoint_csv)
        print(f"   🔄 Resume {app_name} rating={score}: {len(df_existing)} baris sudah ada")
        return state.get("continuation_token"), df_existing.to_dict("records")

    return None, []


def _save_checkpoint(app_name, score, continuation_token, collected):
    os.makedirs(os.path.join(RAW_DIR, "_checkpoints"), exist_ok=True)
    state_path = _state_path(app_name, score)
    checkpoint_csv = os.path.join(RAW_DIR, "_checkpoints", f"{app_name}_score{score}.csv")

    with open(state_path, "wb") as f:
        pickle.dump({"continuation_token": continuation_token, "collected": len(collected)}, f)
    pd.DataFrame(collected).to_csv(checkpoint_csv, index=False)


# ==========================================================
# SCRAPE SATU (app, rating) -- dengan filter bahasa inline
# ==========================================================
def scrape_one_score(app_id, app_name, score, target_count, lang="id", country="id"):
    print(f"\n📡 {app_name} | rating={score} | target={target_count} (setelah filter bahasa)")

    continuation_token, collected = _load_state(app_name, score)
    checkpoint_marker = len(collected)

    n_checked = len(collected)      # total baris yang PERNAH diperiksa (lolos + tidak)
    n_lang_dropped = 0              # dibuang karena bukan/tidak terdeteksi Indonesia
    n_too_short_for_detect = 0      # terlalu pendek untuk dideteksi -> tetap disimpan,
                                     # ditandai lang='undetected', dilaporkan terpisah

    while len(collected) < target_count:
        fetch_count = min(BATCH_SIZE, max(BATCH_SIZE, target_count - len(collected)))
        # catatan: fetch_count sengaja tidak diperkecil ketat mengikuti sisa
        # kuota, karena sebagian hasil akan gugur di filter bahasa -- kalau
        # fetch persis pas sisa kuota, butuh lebih banyak putaran while untuk
        # mencapai target. BATCH_SIZE tetap dipakai sebagai ukuran batch API.

        def fetch():
            return reviews(
                app_id=app_id, lang=lang, country=country,
                sort=Sort.NEWEST, count=BATCH_SIZE,
                filter_score_with=score, continuation_token=continuation_token,
            )

        try:
            result, new_token = _safe_scrape_call(fetch, app_name, score)
        except RuntimeError as e:
            print(f"   ❌ {e}. Lanjut dengan data yang sudah terkumpul.")
            break

        if not result:
            print(f"   Tidak ada ulasan lagi rating={score} di {app_name} "
                  f"(terkumpul {len(collected)}/{target_count}).")
            break

        for r in result:
            n_checked += 1
            text = r.get("content", "") or ""
            detected = _detect_language(text)

            if detected is None:
                n_too_short_for_detect += 1
                keep = True  # teks terlalu pendek untuk dinilai -> tidak digugurkan
                             # otomatis (biar tidak salah buang ulasan pendek yang
                             # valid, mis. "bagus", "oke banget"), tapi ditandai
            elif detected == ACCEPTED_LANG:
                keep = True
            else:
                keep = False
                n_lang_dropped += 1

            if not keep:
                continue

            collected.append({
                "source_app": app_name,
                "review_text": text,
                "rating": r.get("score", score),
                "date": r.get("at"),
                "helpful_votes": r.get("thumbsUpCount", 0),
                "review_id": r.get("reviewId", ""),
                "detected_lang": detected if detected is not None else "undetected_short",
                "scraped_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

        continuation_token = new_token
        print(f"   +{len(result)} diperiksa, {len(collected)}/{target_count} lolos filter bahasa "
              f"(dibuang bahasa lain: {n_lang_dropped})")

        if len(collected) - checkpoint_marker >= 500:
            _save_checkpoint(app_name, score, continuation_token, collected)
            checkpoint_marker = len(collected)
            print(f"   💾 Checkpoint tersimpan ({len(collected)} baris)")

        time.sleep(random.uniform(*SLEEP_RANGE))

        if continuation_token is None:
            print(f"   Halaman habis untuk {app_name} rating={score}.")
            break

    _save_checkpoint(app_name, score, continuation_token, collected)

    df = pd.DataFrame(collected)
    if not df.empty:
        df = df.drop_duplicates(subset=["review_id"])
        df = df[df["review_text"].str.strip() != ""]

    status = "PENUH" if len(df) >= target_count else "KURANG"
    if df.empty:
        status = "KOSONG TOTAL"
        _log_error(app_name, score, "Selesai dengan 0 baris terkumpul -- CEK MANUAL.")

    print(f"✅ Selesai: {app_name} rating={score} -> {len(df)} ulasan bersih "
          f"[{status}] (diperiksa: {n_checked}, dibuang-bahasa: {n_lang_dropped}, "
          f"terlalu-pendek-utk-deteksi: {n_too_short_for_detect})")

    return df, {
        "source_app": app_name, "rating": score, "target": target_count,
        "n_collected": len(df), "n_checked": n_checked,
        "n_lang_dropped": n_lang_dropped,
        "n_too_short_for_detect": n_too_short_for_detect,
        "status": status,
    }


# ==========================================================
# LAPORAN KUALITAS DATA (untuk Bab III/IV) -- dijalankan setelah
# semua scraping selesai, di atas final_df yang sudah dikunci
# ==========================================================
def _report_scraping_summary(scan_summaries):
    df_summary = pd.DataFrame(scan_summaries)
    summary_path = os.path.join(config.RESULTS_DIR, "scraping_summary.csv")
    df_summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print(" RINGKASAN SCRAPING PER (APP, RATING) ")
    print("=" * 60)
    print(df_summary.to_string(index=False))
    print(f"\n💾 Disimpan -> {summary_path}")

    kosong = df_summary[df_summary["status"] == "KOSONG TOTAL"]
    kurang = df_summary[df_summary["status"] == "KURANG"]
    if not kosong.empty:
        print(f"\n🚨 PERINGATAN: {len(kosong)} kombinasi (app, rating) KOSONG TOTAL "
              f"-- cek {ERROR_LOG_FILE} dan pertimbangkan scraping ulang manual.")
    if not kurang.empty:
        print(f"\nℹ️  {len(kurang)} kombinasi (app, rating) tidak mencapai target "
              f"(wajar untuk rating yang secara alami jarang -- laporkan apa adanya).")

    total_checked = df_summary["n_checked"].sum()
    total_lang_dropped = df_summary["n_lang_dropped"].sum()
    pct_dropped = total_lang_dropped / total_checked * 100 if total_checked else 0
    print(f"\n🌐 Filter bahasa keseluruhan: {total_lang_dropped}/{total_checked} "
          f"({pct_dropped:.2f}%) ulasan dibuang karena terdeteksi bukan Bahasa Indonesia.")
    print("   -> Angka ini WAJIB disebutkan di Bab III (langkah pengumpulan data) sebagai")
    print("      bagian dari justifikasi kualitas data, bukan disembunyikan.")


def _report_date_distribution(final_df):
    """
    Cek confound temporal (rating 2/3 vs 1/4/5 berpotensi berasal dari
    rentang waktu berbeda karena kuota lebih kecil = scraper tidak perlu
    menelusuri sejauh rating lain). Tidak diperbaiki di sini (stratified
    sampling tetap valid) -- hanya dilaporkan supaya bisa dibahas eksplisit
    di Bab IV sebagai karakteristik data, sesuai saran pembimbing.
    """
    final_df = final_df.copy()
    final_df["date"] = pd.to_datetime(final_df["date"], errors="coerce")

    date_dist = (
        final_df.groupby(["source_app", "rating"])["date"]
        .agg(["min", "max", "median", "count"])
        .reset_index()
    )
    date_dist_path = os.path.join(config.RESULTS_DIR, "date_distribution_per_app_rating.csv")
    date_dist.to_csv(date_dist_path, index=False)

    print("\n" + "=" * 60)
    print(" DISTRIBUSI TANGGAL PER (APP, RATING) -- cek confound temporal ")
    print("=" * 60)
    print(date_dist.to_string(index=False))
    print(f"\n💾 Disimpan -> {date_dist_path}")
    print("   -> Bandingkan rentang tanggal antar rating per app secara manual.")
    print("      Kalau rating 2/3 rentangnya jauh lebih panjang ke belakang dibanding")
    print("      rating 1/4/5, itu confound temporal yang perlu disebutkan di Bab IV")
    print("      sebagai limitasi/karakteristik data (bukan wajib diperbaiki).")


def _report_text_length(final_df):
    word_counts = final_df["review_text"].astype(str).str.split().apply(len)
    pct_very_short = (word_counts <= 3).mean() * 100

    print("\n" + "=" * 60)
    print(" DISTRIBUSI PANJANG TEKS ULASAN (dalam kata) ")
    print("=" * 60)
    print(word_counts.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_string())
    print(f"\nUlasan sangat pendek (<=3 kata): {pct_very_short:.2f}%")
    print("   -> Ulasan pendek TIDAK dibuang otomatis (kurang informasi != label")
    print("      noise), tapi angka ini perlu dicatat di Bab IV supaya deteksi noise")
    print("      pada ulasan pendek tidak salah ditafsirkan sebagai mismatch rating-teks.")

    length_summary = word_counts.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    length_summary.to_csv(os.path.join(config.RESULTS_DIR, "text_length_summary.csv"))


def _report_duplicate_texts(final_df, top_n=20):
    """
    Deteksi template/spam via exact-text duplicate count (bukan review_id --
    review_id sudah pasti unik lewat dedup sebelumnya). Ini hanya menangkap
    duplikat PERSIS SAMA, bukan near-duplicate dengan variasi kecil (typo,
    tambahan spasi, dst.) -- keterbatasan ini perlu disebutkan kalau dipakai
    sebagai bukti di Bab IV. Tidak difilter otomatis, sesuai saran pembimbing.
    """
    text_counts = final_df["review_text"].value_counts()
    dup_counts = text_counts[text_counts > 1]

    total_rows = len(final_df)
    total_dup_rows = dup_counts.sum()
    pct_dup_rows = total_dup_rows / total_rows * 100 if total_rows else 0

    print("\n" + "=" * 60)
    print(" DETEKSI TEKS DUPLIKAT PERSIS (indikasi template/spam) ")
    print("=" * 60)
    print(f"Baris yang teksnya persis sama dengan >=1 baris lain: "
          f"{total_dup_rows} ({pct_dup_rows:.2f}%)")
    if not dup_counts.empty:
        print(f"\nTop {min(top_n, len(dup_counts))} teks paling sering berulang:")
        print(dup_counts.head(top_n).to_string())

    dup_path = os.path.join(config.RESULTS_DIR, "duplicate_review_texts.csv")
    dup_counts.head(200).to_csv(dup_path, header=["count"])
    print(f"\n💾 Disimpan -> {dup_path}")
    print("   -> Ini HANYA duplikat persis sama (bukan near-duplicate/typo-variant).")
    print("      Kalau porsinya besar, pertimbangkan cek manual di Bab IV apakah ada")
    print("      pola bot/template yang perlu dibahas sebagai limitasi data.")


# ==========================================================
# MAIN
# ==========================================================
def main():
    print("=" * 60)
    print(" GOOGLE PLAY SCRAPER — STRATIFIED PER RATING + FILTER BAHASA ")
    print("=" * 60)

    all_data = []
    scan_summaries = []

    for app_cfg in APPS_CONFIG:
        app_frames = []
        for score, target in TARGET_PER_SCORE.items():
            df_score, summary = scrape_one_score(
                app_id=app_cfg["app_id"], app_name=app_cfg["name"],
                score=score, target_count=target,
            )
            app_frames.append(df_score)
            scan_summaries.append(summary)

        if app_frames:
            df_app = pd.concat(app_frames, ignore_index=True)
            app_path = os.path.join(RAW_DIR, f"{app_cfg['name']}_reviews.csv")
            df_app.to_csv(app_path, index=False)
            print(f"💾 {app_cfg['name']} -> {app_path}")
            all_data.append(df_app)

    if not all_data:
        print("⚠️ Tidak ada data terkumpul.")
        return

    final_df = pd.concat(all_data, ignore_index=True)

    # Dedup final di level gabungan -- sebelumnya hanya dedup per (app, rating)
    # di scrape_one_score; ini jaga-jaga kalau review_id sama pernah lolos
    # kuota dua rating berbeda (kasus langka, tapi murah untuk dicegah).
    n_before_final_dedup = len(final_df)
    final_df = final_df.drop_duplicates(subset=["review_id"])
    n_dropped_final_dedup = n_before_final_dedup - len(final_df)

    final_df["date"] = pd.to_datetime(final_df["date"]).dt.strftime("%Y-%m-%d")
    final_df.to_csv(config.RAW_DATA_FILE, index=False)

    print("\n" + "=" * 60)
    print(f"🎉 SELESAI! Total ulasan (setelah dedup final): {len(final_df)} "
          f"(dibuang saat dedup final: {n_dropped_final_dedup})")
    print(f"📁 File master (LOCKED, jangan diubah lagi): {config.RAW_DATA_FILE}")
    print("\n📊 Distribusi rating keseluruhan:")
    print(final_df["rating"].value_counts().sort_index())
    print("\n📈 Distribusi rating per aplikasi:")
    print(final_df.groupby(["source_app", "rating"]).size().unstack(fill_value=0))

    _report_scraping_summary(scan_summaries)
    _report_date_distribution(final_df)
    _report_text_length(final_df)
    _report_duplicate_texts(final_df)

    print("\n" + "=" * 60)
    print("📝 CATATAN UNTUK BAB 3 (Batasan/Langkah Penelitian):")
    print(f"   Data diambil pada: {datetime.now().strftime('%Y-%m-%d')}")
    print("   - Kalau target rating 2/3 tidak tercapai, itu wajar (rating tengah")
    print("     memang lebih jarang muncul secara alami) -- laporkan apa adanya.")
    print("   - Sertakan angka filter bahasa (lihat scraping_summary.csv) sebagai")
    print("     bagian dari deskripsi langkah pengumpulan data.")
    print("   - Sertakan distribusi tanggal per (app, rating) di Bab IV sebagai")
    print("     pembahasan karakteristik/limitasi data (confound temporal).")
    print("=" * 60)


if __name__ == "__main__":
    main()