import os
import numpy as np
import pandas as pd
import cleanlab
from sklearn.model_selection import train_test_split

from src import config
from src.metrics import compute_metrics
from src.proxy import get_proxy_pred_probs   # <-- SATU-SATUNYA titik integrasi dgn proxy.py


# ==========================================================
# 1. SAMPLING VALIDASI MANUSIA — STRATIFIED BY rating_diff
# ==========================================================
def _stratified_human_sample(df_noise, n, seed=42):
    """
    Stratified by rating_diff (jarak ordinal label vs prediksi proxy), supaya
    sample tidak didominasi kasus ambigu ringan (diff=1) saja -- kasus noise
    parah (diff besar) juga harus terwakili proporsional.

    Fallback: bin rating_diff dengan anggota <2 (tidak bisa distratify oleh
    sklearn) digabung ke leftover pool dan diambil random terpisah, supaya
    tidak error saat data sedikit.
    """
    n = min(n, len(df_noise))
    df = df_noise.copy()
    strat_key = df["rating_diff"].clip(upper=3)  # gabung diff>=3 jadi satu bin

    counts = strat_key.value_counts()
    valid_bins = counts[counts >= 2].index
    df_stratifiable = df[strat_key.isin(valid_bins)]
    df_leftover = df[~strat_key.isin(valid_bins)]

    if len(df_stratifiable) >= n:
        sample, _ = train_test_split(
            df_stratifiable, train_size=n, random_state=seed,
            stratify=strat_key.loc[df_stratifiable.index],
        )
    else:
        needed = n - len(df_stratifiable)
        extra = df_leftover.sample(n=min(needed, len(df_leftover)), random_state=seed)
        sample = pd.concat([df_stratifiable, extra])

    return sample.sample(frac=1, random_state=seed)


def export_human_validation_sample(df_noise, n=None):
    n = n or config.HUMAN_VALIDATION_N
    sample = _stratified_human_sample(df_noise, n)
    sample = sample[[
        "source_app", "review_text", "cleaned_text",
        "rating", "predicted_rating", "rating_diff",
    ]].copy()
    sample["human_verdict"] = ""
    sample["human_note"] = ""
    sample.to_csv(config.HUMAN_VALIDATION_FILE, index=False)

    print(f"\n📝 Sample validasi manusia ({len(sample)} baris, stratified by rating_diff):")
    print(f"   {config.HUMAN_VALIDATION_FILE}")
    print(f"   Distribusi: {sample['rating_diff'].value_counts().sort_index().to_dict()}")
    print("   -> Isi kolom 'human_verdict' manual, lalu jalankan src.human_validation.compute_agreement()")


# ==========================================================
# 2. LOG KUALITAS PROXY — akumulatif, idempotent per (proxy_id, data_version)
# ==========================================================
def _log_proxy_quality(proxy_acc, proxy_metrics, pct_flagged):
    """
    Satu baris per (proxy_id, data_version) di results/proxy_ablation_table.csv.
    Idempotent: kalau clean.py dijalankan ulang dengan kombinasi proxy+data
    yang sama, baris lama diganti -- bukan duplikat. Kombinasi BERBEDA
    (mis. P4 di data v1 vs v2) tetap berdampingan, tidak saling menimpa --
    ini yang sebelumnya bug karena kunci dedup cuma proxy_id.
    """
    row = {
        "proxy_id": config.PROXY_ID,
        "proxy_name": config.PROXY_NAME,
        "proxy_desc": config.PROXY_DESC,
        "data_version": config.DATA_VERSION,
        "accuracy": proxy_acc,
        "mae": proxy_metrics["mae"],
        "off_by_one": proxy_metrics["off_by_one"],
        "qwk": proxy_metrics["qwk"],
        "pct_flagged_noise": pct_flagged,
    }
    log_df = pd.DataFrame([row])

    if os.path.exists(config.PROXY_QUALITY_LOG_FILE):
        existing = pd.read_csv(config.PROXY_QUALITY_LOG_FILE)
        if "data_version" not in existing.columns:
            existing["data_version"] = "v1"  # baris lama sebelum kolom ini ada
        existing = existing[
            ~((existing["proxy_id"] == config.PROXY_ID) &
              (existing["data_version"] == config.DATA_VERSION))
        ]
        log_df = pd.concat([existing, log_df], ignore_index=True).sort_values(
            ["data_version", "proxy_id"]
        )

    log_df.to_csv(config.PROXY_QUALITY_LOG_FILE, index=False)
    print(f"📄 Tabel ablasi proxy diperbarui -> {config.PROXY_QUALITY_LOG_FILE}")


# ==========================================================
# 3. PIPELINE UTAMA CONFIDENT LEARNING
# ==========================================================
def run_confident_learning():
    print("=" * 60)
    print(f" CONFIDENT LEARNING — proxy aktif: [{config.PROXY_ID}] {config.PROXY_NAME} "
          f"| data: {config.DATA_VERSION} ")
    print("=" * 60)

    df_train = pd.read_csv(config.TRAIN_RAW_FILE)

    if config.DEBUG_MODE and len(df_train) > config.DEBUG_SAMPLE_SIZE:
        df_train, _ = train_test_split(
            df_train, train_size=config.DEBUG_SAMPLE_SIZE,
            random_state=42, stratify=df_train["rating"],
        )
        print(f"[DEBUG] subset -> {len(df_train)} baris")

    print(f"📥 Memuat {len(df_train)} baris data train.")
    labels = df_train["rating"].values - 1  # 0-indexed
    texts = df_train["cleaned_text"].tolist()

    pred_probs = get_proxy_pred_probs(texts, labels)

    if pred_probs.shape != (len(texts), config.NUM_CLASSES):
        raise ValueError(
            f"Bentuk pred_probs dari proxy [{config.PROXY_NAME}] tidak sesuai: "
            f"dapat {pred_probs.shape}, diharapkan ({len(texts)}, {config.NUM_CLASSES}). "
            f"Cek implementasi proxy ini di src/proxy.py."
        )

    proxy_preds = np.argmax(pred_probs, axis=1)
    proxy_acc = (proxy_preds == labels).mean()
    proxy_metrics = compute_metrics(labels, proxy_preds)

    print(f"\n📐 Kualitas proxy [{config.PROXY_NAME}] (data {config.DATA_VERSION}):")
    print(f"   Exact Accuracy : {proxy_acc:.4f}")
    print(f"   MAE            : {proxy_metrics['mae']:.4f}")
    print(f"   Off-by-1 Acc   : {proxy_metrics['off_by_one']:.4f}")
    print(f"   QWK            : {proxy_metrics['qwk']:.4f}")

    df_train = df_train.copy()
    df_train["predicted_rating"] = proxy_preds + 1
    df_train["rating_diff"] = (df_train["rating"] - df_train["predicted_rating"]).abs()

    results = {}
    print("\n🔎 Analisis Metode Filter Cleanlab:")
    for method in config.CLEANLAB_FILTER_METHODS:
        issues = cleanlab.filter.find_label_issues(labels=labels, pred_probs=pred_probs, filter_by=method)
        results[method] = issues
        print(f"   '{method}': {issues.sum()} baris diflag ({issues.sum()/len(df_train)*100:.2f}%)")

    main_method = "confident_learning"
    df_train["is_noise"] = results[main_method]
    for method, issues in results.items():
        df_train[f"is_noise__{method}"] = issues

    df_train["is_noise_severe"] = df_train["is_noise"] & (df_train["rating_diff"] >= config.SEVERITY_THRESHOLD)

    df_cleaned_hard = df_train[~df_train["is_noise"]].copy()
    df_cleaned_severe = df_train[~df_train["is_noise_severe"]].copy()
    df_noise = df_train[df_train["is_noise"]].copy()

    print(f"\n✅ Deteksi selesai (metode utama: {main_method}).")
    print(f"   Hard-prune     : buang {len(df_noise)} / sisa {len(df_cleaned_hard)}")
    print(f"   Severity-aware : buang {df_train['is_noise_severe'].sum()} / sisa {len(df_cleaned_severe)}")
    print("\n📊 Distribusi rating_diff pada baris noise:")
    print(df_noise["rating_diff"].value_counts().sort_index())

    _log_proxy_quality(proxy_acc, proxy_metrics, pct_flagged=len(df_noise) / len(df_train) * 100)

    drop_cols = [c for c in df_cleaned_hard.columns
                 if c.startswith("is_noise") or c in ("predicted_rating", "rating_diff")]
    df_cleaned_hard.drop(columns=drop_cols, errors="ignore").to_csv(config.TRAIN_CLEANED_HARD_FILE, index=False)
    df_cleaned_severe.drop(columns=drop_cols, errors="ignore").to_csv(config.TRAIN_CLEANED_SEVERE_FILE, index=False)
    print(f"\n💾 Cleaned (hard)   -> {config.TRAIN_CLEANED_HARD_FILE}")
    print(f"💾 Cleaned (severe) -> {config.TRAIN_CLEANED_SEVERE_FILE}")

    df_noise.to_csv(config.NOISE_SAMPLES_FILE, index=False)
    export_human_validation_sample(df_noise)

    return df_noise, proxy_metrics


if __name__ == "__main__":
    run_confident_learning()