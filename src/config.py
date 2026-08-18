import os
import sys
import torch

# ==========================================================
# 1. DETEKSI LINGKUNGAN & PATH DASAR
# ==========================================================
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    DRIVE_ROOT = '/content/drive/MyDrive/SKRIPSI_CORN'
else:
    DRIVE_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "local_data")

DATA_RAW_DIR = os.path.join(DRIVE_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(DRIVE_ROOT, "data", "processed")
PROXY_CACHE_DIR = os.path.join(DRIVE_ROOT, "proxy_cache")
CLEANED_DIR = os.path.join(DRIVE_ROOT, "cleaned")
MODEL_CKPT_ROOT = os.path.join(DRIVE_ROOT, "models_ckpt")
HUMAN_VALIDATION_DIR = os.path.join(DRIVE_ROOT, "human_validation")
RESULTS_DIR = os.path.join(DRIVE_ROOT, "results")
LOGS_DIR = os.path.join(DRIVE_ROOT, "logs")

for d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, PROXY_CACHE_DIR, CLEANED_DIR,
          MODEL_CKPT_ROOT, HUMAN_VALIDATION_DIR, RESULTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# ==========================================================
# 2. FILE DATA -- versioned via set_data_version(), BUKAN assignment langsung
# ==========================================================
RAW_DATA_FILE = os.path.join(DATA_RAW_DIR, "all_reviews_master.csv")  # sumber mentah tunggal, tidak versioned

DATA_VERSION = "v2"

def set_data_version(version):
    """
    WAJIB dipanggil untuk ganti versi data di tengah sesi -- assignment
    langsung `config.DATA_VERSION = "v1"` TIDAK mengubah CLEAN_TEXT_FILE/
    TRAIN_RAW_FILE/TEST_FILE, karena path-path itu perlu dihitung ulang
    setiap kali versi berubah, tidak cukup sekali saat modul di-import.
    """
    global DATA_VERSION, CLEAN_TEXT_FILE, TRAIN_RAW_FILE, TEST_FILE
    DATA_VERSION = version
    CLEAN_TEXT_FILE = os.path.join(DATA_PROCESSED_DIR, f"reviews_clean__{DATA_VERSION}.csv")
    TRAIN_RAW_FILE = os.path.join(DATA_PROCESSED_DIR, f"split_train_raw__{DATA_VERSION}.csv")
    TEST_FILE = os.path.join(DATA_PROCESSED_DIR, f"split_test__{DATA_VERSION}.csv")
    print(f"📁 Data version aktif: {DATA_VERSION}")

set_data_version(DATA_VERSION)  # populate path pertama kali saat modul di-import

# ==========================================================
# 3. REGISTRY PROXY CLASSIFIER — GANTI PROXY DENGAN set_proxy(id)
# ==========================================================
PROXY_REGISTRY = {
    0: {"name": "frozen_cls_lr", "desc": "CLS embedding beku + LR (P1)",
        "backbone": "indobenchmark/indobert-base-p1"},
    1: {"name": "frozen_meanpool_lr", "desc": "Mean-pool embedding beku + LR (P2)",
        "backbone": "indobenchmark/indobert-base-p1"},
    2: {"name": "finetuned_ce", "desc": "IndoBERT fine-tuned K-Fold, CE loss (P3)",
        "backbone": "indobenchmark/indobert-base-p1"},
    3: {"name": "finetuned_corn", "desc": "IndoBERT fine-tuned K-Fold, CORN loss (P4) -- DEFAULT/FINAL",
        "backbone": "indobenchmark/indobert-base-p1"},
    4: {"name": "finetuned_corn_fusion", "desc": "IndoBERT+CORN + fusi sentimen (P5)",
        "backbone": "indobenchmark/indobert-base-p1"},
    6: {"name": "finetuned_corn_indobertweet",
        "desc": "IndoBERTweet fine-tuned K-Fold, CORN loss (P6)",
        "backbone": "indolem/indobertweet-base-uncased"},
}

def set_proxy(proxy_id):
    """
    Ganti proxy aktif dengan AMAN di tengah sesi Python -- TANPA
    importlib.reload(config). Pakai: config.set_proxy(pid)
    JANGAN pakai: config.PROXY_ID = pid lalu importlib.reload(config)
    """
    global PROXY_ID, PROXY_NAME, PROXY_DESC, PRETRAINED_MODEL_NAME
    global PROXY_PRED_PROBS_FILE, PROXY_PRED_PROBS_META_FILE
    global TRAIN_CLEANED_HARD_FILE, TRAIN_CLEANED_SEVERE_FILE, MODEL_CKPT_DIR
    global HUMAN_VALIDATION_FILE, HUMAN_VALIDATION_RESULT_FILE
    global PROGRESS_FILE, NOISE_SAMPLES_FILE
    global FINAL_RESULTS_TABLE_FILE, SIGNIFICANCE_TEST_FILE

    if proxy_id not in PROXY_REGISTRY:
        raise ValueError(f"PROXY_ID tidak dikenal: {proxy_id}")

    entry = PROXY_REGISTRY[proxy_id]
    PROXY_ID = proxy_id
    PROXY_NAME = entry["name"]
    PROXY_DESC = entry["desc"]
    PRETRAINED_MODEL_NAME = entry["backbone"]

    tag = f"{PROXY_NAME}__{DATA_VERSION}"  # kunci gabungan proxy+data yang dipakai di semua path turunan

    PROXY_PRED_PROBS_FILE = os.path.join(PROXY_CACHE_DIR, f"oof_pred_probs__{tag}.npy")
    PROXY_PRED_PROBS_META_FILE = os.path.join(PROXY_CACHE_DIR, f"oof_pred_probs_meta__{tag}.csv")
    TRAIN_CLEANED_HARD_FILE = os.path.join(CLEANED_DIR, f"train_cleaned_hard__{tag}.csv")
    TRAIN_CLEANED_SEVERE_FILE = os.path.join(CLEANED_DIR, f"train_cleaned_severe__{tag}.csv")
    MODEL_CKPT_DIR = os.path.join(MODEL_CKPT_ROOT, tag)
    os.makedirs(MODEL_CKPT_DIR, exist_ok=True)

    HUMAN_VALIDATION_FILE = os.path.join(HUMAN_VALIDATION_DIR, f"human_validation_sample__{tag}.csv")
    HUMAN_VALIDATION_RESULT_FILE = os.path.join(HUMAN_VALIDATION_DIR, f"human_validation_result__{tag}.csv")

    # FIX: sebelumnya statis -- resume Cell 8 dan noise-sample bisa salah
    # timpa/skip kalau kamu ganti proxy atau data version tanpa restart.
    PROGRESS_FILE = os.path.join(LOGS_DIR, f"experiment_progress__{tag}.json")
    NOISE_SAMPLES_FILE = os.path.join(DATA_PROCESSED_DIR, f"detected_noise_samples__{tag}.csv")

    # FIX: hasil akhir Bab 4 sekarang per backbone+data -- retrain dengan
    # kombinasi berbeda tidak lagi menimpa final_results_table.csv yang sudah jadi.
    FINAL_RESULTS_TABLE_FILE = os.path.join(RESULTS_DIR, f"final_results_table__{tag}.csv")
    SIGNIFICANCE_TEST_FILE = os.path.join(RESULTS_DIR, f"significance_test__{tag}.csv")

    print(f"📌 Proxy aktif: [{PROXY_ID}] {PROXY_NAME} — {PROXY_DESC}")
    print(f"   Backbone: {PRETRAINED_MODEL_NAME} | Data: {DATA_VERSION}")

EMBEDDING_CACHE_FILE = os.path.join(PROXY_CACHE_DIR, "embeddings_meanpool.npy")
EMBEDDING_CACHE_META_FILE = os.path.join(PROXY_CACHE_DIR, "embeddings_meanpool_meta.csv")

PROXY_QUALITY_LOG_FILE = os.path.join(RESULTS_DIR, "proxy_ablation_table.csv")  # tabel akumulatif, tetap satu file

set_proxy(3)   # default saat modul di-import pertama kali

# ==========================================================
# 4. MODEL & HYPERPARAMETER (SAMA UNTUK SEMUA SKENARIO)
# ==========================================================
SENTIMENT_MODEL_NAME = "w11wo/indonesian-roberta-base-sentiment-classifier"  # dipakai hanya jika PROXY_ID == 4

MAX_LEN = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_CLASSES = 5

EPOCHS = 8
SEED_LIST = [42, 123, 2024]
PATIENCE = 3

PROXY_CV_FOLDS = 5
PROXY_FINETUNE_EPOCHS = 3
PROXY_FINETUNE_LR = 2e-5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# 5. FAST DEV MODE (hanya untuk uji kode, JANGAN dipakai laporan)
# ==========================================================
DEBUG_MODE = os.environ.get("SKRIPSI_DEBUG", "0") == "1"
DEBUG_SAMPLE_SIZE = 600
DEBUG_EPOCHS = 2
DEBUG_CV_FOLDS = 2
DEBUG_SEED_LIST = [42]

if DEBUG_MODE:
    EPOCHS = DEBUG_EPOCHS
    PROXY_CV_FOLDS = DEBUG_CV_FOLDS
    SEED_LIST = DEBUG_SEED_LIST
    print(f"⚠️  SKRIPSI_DEBUG=1 AKTIF — subset {DEBUG_SAMPLE_SIZE} baris, {EPOCHS} epoch, seed {SEED_LIST}")
    print("    HASIL DI MODE INI TIDAK BOLEH DIPAKAI UNTUK LAPORAN.")

# ==========================================================
# 6. SPLIT VALIDASI, CLEANING, SEVERITY
# ==========================================================
VAL_SIZE = 0.1
CLEANLAB_FILTER_METHODS = ["confident_learning", "prune_by_noise_rate"]
SEVERITY_THRESHOLD = 2

# ==========================================================
# 7. VALIDASI MANUSIA
# ==========================================================
HUMAN_VALIDATION_N = 50