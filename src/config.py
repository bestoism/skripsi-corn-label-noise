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
    # Untuk uji coba/debug singkat di lokal (VSCode), tanpa Drive
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
# 2. FILE DATA (SPLIT DIKUNCI — JANGAN DIGENERATE ULANG)
# ==========================================================
RAW_DATA_FILE = os.path.join(DATA_RAW_DIR, "all_reviews_master.csv")
CLEAN_TEXT_FILE = os.path.join(DATA_PROCESSED_DIR, "reviews_clean.csv")
TRAIN_RAW_FILE = os.path.join(DATA_PROCESSED_DIR, "split_train_raw.csv")
TEST_FILE = os.path.join(DATA_PROCESSED_DIR, "split_test.csv")

# ==========================================================
# 3. REGISTRY PROXY CLASSIFIER — GANTI PROXY DI SINI SAJA
# ==========================================================
# Cukup ubah angka PROXY_ID di bawah untuk pindah metode proxy.
# Semua path cache/model/hasil otomatis mengikuti nama metode ini,
# jadi tidak akan pernah saling menimpa antar metode.
PROXY_REGISTRY = {
    0: {"name": "frozen_cls_lr",        "desc": "CLS embedding beku + Logistic Regression (P1)"},
    1: {"name": "frozen_meanpool_lr",   "desc": "Mean-pooling embedding beku + Logistic Regression (P2)"},
    2: {"name": "finetuned_ce",         "desc": "IndoBERT fine-tuned K-Fold, CE loss (P3)"},
    3: {"name": "finetuned_corn",       "desc": "IndoBERT fine-tuned K-Fold, CORN loss (P4) -- DEFAULT/FINAL"},
    4: {"name": "finetuned_corn_fusion","desc": "IndoBERT+CORN + fusi sentimen eksternal (P5)"},
}

PROXY_ID = 3   # <--- UBAH ANGKA INI SAJA UNTUK GANTI PROXY (0-4)

PROXY_NAME = PROXY_REGISTRY[PROXY_ID]["name"]
PROXY_DESC = PROXY_REGISTRY[PROXY_ID]["desc"]

# Path proxy & hasil cleaning otomatis ter-suffix nama proxy aktif
PROXY_PRED_PROBS_FILE = os.path.join(PROXY_CACHE_DIR, f"oof_pred_probs__{PROXY_NAME}.npy")
PROXY_PRED_PROBS_META_FILE = os.path.join(PROXY_CACHE_DIR, f"oof_pred_probs_meta__{PROXY_NAME}.csv")
EMBEDDING_CACHE_FILE = os.path.join(PROXY_CACHE_DIR, "embeddings_meanpool.npy")
EMBEDDING_CACHE_META_FILE = os.path.join(PROXY_CACHE_DIR, "embeddings_meanpool_meta.csv")

TRAIN_CLEANED_HARD_FILE = os.path.join(CLEANED_DIR, f"train_cleaned_hard__{PROXY_NAME}.csv")
TRAIN_CLEANED_SEVERE_FILE = os.path.join(CLEANED_DIR, f"train_cleaned_severe__{PROXY_NAME}.csv")

# Checkpoint model final JUGA dipisah per proxy aktif, supaya percobaan
# dengan proxy berbeda tidak saling menimpa checkpoint satu sama lain.
MODEL_CKPT_DIR = os.path.join(MODEL_CKPT_ROOT, PROXY_NAME)
os.makedirs(MODEL_CKPT_DIR, exist_ok=True)

# Tabel perbandingan kualitas antar-proxy (dipakai utk pilot study Bab 1)
PROXY_QUALITY_LOG_FILE = os.path.join(RESULTS_DIR, "proxy_ablation_table.csv")

# ==========================================================
# 4. MODEL & HYPERPARAMETER (SAMA UNTUK SEMUA SKENARIO)
# ==========================================================
PRETRAINED_MODEL_NAME = "indobenchmark/indobert-base-p1"
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
HUMAN_VALIDATION_FILE = os.path.join(HUMAN_VALIDATION_DIR, "human_validation_sample.csv")
HUMAN_VALIDATION_RESULT_FILE = os.path.join(HUMAN_VALIDATION_DIR, "human_validation_result.csv")

# ==========================================================
# 8. HASIL AKHIR
# ==========================================================
FINAL_RESULTS_TABLE_FILE = os.path.join(RESULTS_DIR, "final_results_table.csv")
SIGNIFICANCE_TEST_FILE = os.path.join(RESULTS_DIR, "significance_test.csv")
NOISE_SAMPLES_FILE = os.path.join(DATA_PROCESSED_DIR, "detected_noise_samples.csv")
PROGRESS_FILE = os.path.join(LOGS_DIR, "experiment_progress.json")

print(f"📌 Proxy aktif: [{PROXY_ID}] {PROXY_NAME} — {PROXY_DESC}")