"""
Scraper Google Play Store, stratified per rating (1-5), dengan checkpoint
yang benar-benar resumable dan retry untuk koneksi tidak stabil.

Dijalankan SEKALI di awal (bukan bagian pipeline utama) -- setelah selesai,
hasilnya (all_reviews_master.csv) dikunci dan dipakai sebagai sumber
kebenaran tunggal untuk seluruh eksperimen berikutnya.

Cara pakai (di Colab, setelah mount Drive & clone repo):
    from scripts.scrape_google_play import main
    main()
"""

import os
import json
import pickle  # tambahkan di bagian import paling atas file
import time
import random
from datetime import datetime

import pandas as pd
from google_play_scraper import reviews, Sort

from src import config

# ==========================================================
# KONFIGURASI SCRAPING
# ==========================================================
APPS_CONFIG = [
    {"app_id": "id.co.bankbkemobile.digitalbank", "name": "SeaBank"},
    {"app_id": "com.tokopedia.tkpd", "name": "Tokopedia"},
    {"app_id": "com.gojek.app", "name": "Gojek"},
]

# Target per rating -- rating tengah (2,3) sengaja ditarget lebih kecil
# karena secara alami lebih jarang muncul; JANGAN dipaksa seimbang dengan
# menaikkan target-nya, itu hanya akan membuat scraper gagal terus tanpa
# hasil (lihat catatan di akhir main()).
TARGET_PER_SCORE = {1: 800, 2: 500, 3: 500, 4: 700, 5: 800}

BATCH_SIZE = 200
MAX_RETRIES = 5
SLEEP_RANGE = (2, 5)

RAW_DIR = config.DATA_RAW_DIR
STATE_DIR = os.path.join(config.DRIVE_ROOT, "scrape_state")
os.makedirs(STATE_DIR, exist_ok=True)


# ==========================================================
# RETRY HELPER
# ==========================================================
def _safe_scrape_call(fetch_fn, max_retries=MAX_RETRIES):
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
            time.sleep(wait)
    raise RuntimeError(f"Gagal setelah {max_retries}x percobaan: {last_error}")


# ==========================================================
# CHECKPOINT — BENAR-BENAR RESUMABLE (fix: pakai pickle, bukan json,
# karena continuation_token dari google_play_scraper adalah object
# custom _ContinuationToken, bukan tipe data primitif yang bisa
# di-JSON-kan langsung)
# ==========================================================
def _state_path(app_name, score):
    return os.path.join(STATE_DIR, f"{app_name}_score{score}.pkl")  # <-- .pkl, bukan .json


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
# SCRAPE SATU (app, rating)
# ==========================================================
def scrape_one_score(app_id, app_name, score, target_count, lang="id", country="id"):
    print(f"\n📡 {app_name} | rating={score} | target={target_count}")

    continuation_token, collected = _load_state(app_name, score)
    checkpoint_marker = len(collected)

    while len(collected) < target_count:
        fetch_count = min(BATCH_SIZE, target_count - len(collected))

        def fetch():
            return reviews(
                app_id=app_id, lang=lang, country=country,
                sort=Sort.NEWEST, count=fetch_count,
                filter_score_with=score, continuation_token=continuation_token,
            )

        try:
            result, new_token = _safe_scrape_call(fetch)
        except RuntimeError as e:
            print(f"   ❌ {e}. Lanjut dengan data yang sudah terkumpul.")
            break

        if not result:
            print(f"   Tidak ada ulasan lagi rating={score} di {app_name} "
                  f"(terkumpul {len(collected)}/{target_count}).")
            break

        for r in result:
            collected.append({
                "source_app": app_name,
                "review_text": r.get("content", ""),
                "rating": r.get("score", score),
                "date": r.get("at"),
                "helpful_votes": r.get("thumbsUpCount", 0),
                "review_id": r.get("reviewId", ""),
                "scraped_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

        continuation_token = new_token
        print(f"   +{len(result)} ulasan (total: {len(collected)}/{target_count})")

        if len(collected) - checkpoint_marker >= 500:
            _save_checkpoint(app_name, score, continuation_token, collected)
            checkpoint_marker = len(collected)
            print(f"   💾 Checkpoint tersimpan ({len(collected)} baris)")

        time.sleep(random.uniform(*SLEEP_RANGE))

        if continuation_token is None:
            print(f"   Halaman habis untuk {app_name} rating={score}.")
            break

    _save_checkpoint(app_name, score, continuation_token, collected)  # simpan progres terakhir

    df = pd.DataFrame(collected)
    if not df.empty:
        df = df.drop_duplicates(subset=["review_id"])
        df = df[df["review_text"].str.strip() != ""]

    print(f"✅ Selesai: {app_name} rating={score} -> {len(df)} ulasan bersih.")
    return df


# ==========================================================
# MAIN
# ==========================================================
def main():
    print("=" * 60)
    print(" GOOGLE PLAY SCRAPER — STRATIFIED PER RATING ")
    print("=" * 60)

    all_data = []
    for app_cfg in APPS_CONFIG:
        app_frames = []
        for score, target in TARGET_PER_SCORE.items():
            df_score = scrape_one_score(
                app_id=app_cfg["app_id"], app_name=app_cfg["name"],
                score=score, target_count=target,
            )
            app_frames.append(df_score)

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
    final_df["date"] = pd.to_datetime(final_df["date"]).dt.strftime("%Y-%m-%d")
    final_df.to_csv(config.RAW_DATA_FILE, index=False)

    print("\n" + "=" * 60)
    print(f"🎉 SELESAI! Total ulasan: {len(final_df)}")
    print(f"📁 File master (LOCKED, jangan diubah lagi): {config.RAW_DATA_FILE}")
    print("\n📊 Distribusi rating keseluruhan:")
    print(final_df["rating"].value_counts().sort_index())
    print("\n📈 Distribusi rating per aplikasi:")
    print(final_df.groupby(["source_app", "rating"]).size().unstack(fill_value=0))
    print("=" * 60)
    print("\n📝 CATATAN UNTUK BAB 3 (Batasan Penelitian):")
    print(f"   Data diambil pada: {datetime.now().strftime('%Y-%m-%d')}")
    print("   Kalau target rating 2/3 tidak tercapai, itu wajar (rating tengah")
    print("   memang lebih jarang muncul secara alami) -- laporkan apa adanya,")
    print("   jangan paksakan seimbang dengan menaikkan target scraping.")


if __name__ == "__main__":
    main()