"""
significance.py -- Uji signifikansi statistik (Wilcoxon Signed-Rank +
Holm-Bonferroni) dan effect size (bootstrap CI 95%) untuk 4 hipotesis
pre-registered (H1-H4), sesuai rumusan masalah Bab 1.3 dan rencana
pengujian Subbab 3.9.2.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from src import config
from src.data import ReviewDataset
from src.models import build_model
from coral_pytorch.dataset import corn_label_from_logits


# ==========================================================
# 0. GUARD -- significance testing HARUS dijalankan dengan proxy final aktif
# ==========================================================
# Checkpoint model final (M1-M6) disimpan di config.MODEL_CKPT_DIR, yang
# path-nya bergantung pada config.PROXY_NAME saat ini (lihat config.set_proxy()).
# Kalau kamu habis eksplorasi proxy lain (P1-P3, P5) di sesi yang sama dan
# lupa config.set_proxy(3) sebelum menjalankan modul ini, fungsi di bawah
# akan mencari checkpoint di folder proxy yang SALAH -- gagalnya jelas
# (FileNotFoundError), tapi assert eksplisit ini membuat penyebabnya
# langsung ketahuan dari pesan error, bukan perlu ditelusuri manual dulu.
_FINAL_PROXY_ID = 3  # finetuned_corn (P4) -- proxy final sesuai Batasan Masalah


def _assert_final_proxy_active():
    if config.PROXY_ID != _FINAL_PROXY_ID:
        raise RuntimeError(
            f"Significance testing (M1-M6) harus dijalankan dengan proxy final "
            f"aktif (PROXY_ID={_FINAL_PROXY_ID}, '{config.PROXY_REGISTRY[_FINAL_PROXY_ID]['name']}'), "
            f"tapi proxy aktif saat ini adalah PROXY_ID={config.PROXY_ID} "
            f"('{config.PROXY_NAME}'). Checkpoint model final (M1-M6) hanya disimpan "
            f"di bawah folder proxy final -- kalau kamu barusan eksplorasi proxy lain "
            f"(P1-P3, P5) di sesi yang sama, jalankan config.set_proxy({_FINAL_PROXY_ID}) "
            f"dulu sebelum memanggil fungsi ini."
        )


# ==========================================================
# 1. KUMPULKAN PREDIKSI DARI KETIGA SEED (bukan cuma seed 42)
# ==========================================================
def _get_predictions_one_seed(scenario_name, loss_type, seed, test_loader):
    model = build_model(loss_type)
    ckpt_path = os.path.join(config.MODEL_CKPT_DIR, scenario_name, f"seed{seed}_best.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint tidak ditemukan: {ckpt_path}\n"
            f"Pastikan run_experiment('{scenario_name}', ..., seed={seed}) "
            f"sudah selesai dijalankan sebelum uji signifikansi, DAN proxy aktif "
            f"saat ini ({config.PROXY_NAME}) sama dengan proxy yang dipakai saat "
            f"training M1-M6 tersebut."
        )

    model.load_state_dict(torch.load(ckpt_path, map_location=config.DEVICE))
    model.eval()

    preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(config.DEVICE)
            attention_mask = batch["attention_mask"].to(config.DEVICE)
            logits = model(input_ids, attention_mask)
            p = (torch.argmax(logits, dim=1) if loss_type == "ce"
                 else corn_label_from_logits(logits)).cpu().numpy() + 1  # kembali ke skala 1-5
            preds.extend(p)

    return np.array(preds)


def collect_all_predictions(scenarios):
    """
    Mengumpulkan prediksi test set dari SEMUA seed (bukan cuma seed 42),
    sesuai Subbab 3.9.2: uji Wilcoxon dihitung dari rata-rata absolute
    error yang diagregasi dari tiga random seed.

    Return:
        true_labels: array label test set asli (skala 1-5)
        preds_per_seed: dict {scenario_name: {seed: array_prediksi}}
    """
    _assert_final_proxy_active()

    df_test = pd.read_csv(config.TEST_FILE)
    test_loader = DataLoader(
        ReviewDataset(df_test["cleaned_text"].tolist(), df_test["rating"].tolist()),
        batch_size=config.BATCH_SIZE, shuffle=False,
    )
    true_labels = np.array(df_test["rating"].tolist())

    preds_per_seed = {}
    for s in scenarios:
        preds_per_seed[s["name"]] = {}
        for seed in config.SEED_LIST:
            print(f"   Memuat prediksi {s['name']} | seed {seed} ...")
            preds_per_seed[s["name"]][seed] = _get_predictions_one_seed(
                s["name"], s["loss"], seed, test_loader
            )

    return true_labels, preds_per_seed


# ==========================================================
# 2. AGREGASI ABSOLUTE ERROR LINTAS 3 SEED
# ==========================================================
def aggregate_errors_across_seeds(true_labels, preds_per_seed):
    """
    Untuk tiap skenario, hitung absolute error per sampel PER SEED, lalu
    rata-ratakan across seed (bukan across sampel!) -- sehingga tiap sampel
    test set punya satu nilai error yang mewakili konsistensi model di
    3 inisialisasi bobot berbeda. Sesuai Subbab 3.9.2: "rata-rata absolute
    error per sampel data uji yang diagregasi dari tiga random seed".

    Return: dict {scenario_name: array shape (n_test_samples,)}
    """
    aggregated = {}
    for scenario_name, seed_preds in preds_per_seed.items():
        errors_per_seed = np.stack([
            np.abs(true_labels - preds) for preds in seed_preds.values()
        ])
        aggregated[scenario_name] = errors_per_seed.mean(axis=0)
    return aggregated


# ==========================================================
# 3. UJI SIGNIFIKANSI — 4 HIPOTESIS PRE-REGISTERED
# ==========================================================
# H4 DITAMBAHKAN (M6 vs M5): rumusan masalah Bab 1.3 secara eksplisit
# menanyakan "Apakah strategi severity-aware pruning ... memberikan manfaat
# performa yang signifikan secara statistik dibandingkan baseline TANPA
# CLEANING DAN DIBANDINGKAN HARD-PRUNING" -- H1-H3 sebelumnya hanya
# menjawab bagian "dibandingkan baseline" (lewat H2: M6 vs M4) dan
# "hard-prune vs baseline" (H3: M5 vs M4), tapi TIDAK PERNAH membandingkan
# M6 vs M5 secara langsung -- padahal itu perbandingan paling langsung
# menjawab "severity-aware LEBIH BAIK DARI hard-pruning atau tidak", bukan
# disimpulkan secara tidak langsung dari selisih H2 dan H3 terhadap
# baseline yang sama (itu bisa menyesatkan -- signifikan/tidaknya H2 dan H3
# terhadap M4 TIDAK OTOMATIS memberi tahu apakah M6 signifikan berbeda dari
# M5 sendiri, karena keduanya dibandingkan ke pembanding yang berbeda arah
# variansnya).
PRE_REGISTERED_HYPOTHESES = [
    ("H1_CORN_vs_CE_raw",         "M4_Baseline_CORN",      "M1_Baseline_CE"),
    ("H2_SeverityAware_vs_Base",  "M6_CleanedSevere_CORN", "M4_Baseline_CORN"),
    ("H3_HardPrune_vs_Base",      "M5_CleanedHard_CORN",   "M4_Baseline_CORN"),
    ("H4_SeverityAware_vs_Hard",  "M6_CleanedSevere_CORN", "M5_CleanedHard_CORN"),
]


def run_significance_test(aggregated_errors, alpha=0.05):
    """
    Wilcoxon Signed-Rank per hipotesis pre-registered (sekarang 4, bukan 3),
    dikoreksi BERSAMA dengan Holm-Bonferroni -- koreksi otomatis menghitung
    SEMUA uji yang didaftarkan di PRE_REGISTERED_HYPOTHESES (multipletests
    menerima daftar p-value apa adanya, jadi menambah H4 ke daftar di atas
    sudah cukup untuk membuat koreksi berbasis 4 uji, tidak perlu ubah kode
    di bawah ini).
    """
    raw_pvalues = []
    rows = []

    for hyp_name, model_a, model_b in PRE_REGISTERED_HYPOTHESES:
        if model_a not in aggregated_errors or model_b not in aggregated_errors:
            raise KeyError(
                f"Skenario '{model_a}' atau '{model_b}' tidak ditemukan di "
                f"aggregated_errors. Pastikan semua 6 skenario sudah dilatih."
            )

        errors_a = aggregated_errors[model_a]
        errors_b = aggregated_errors[model_b]

        # zero_method='zsplit': menangani kasus error_a == error_b persis sama
        # (umum terjadi karena rating diskrit 1-5, banyak sampel error-nya identik)
        stat, p = wilcoxon(errors_a, errors_b, zero_method="zsplit")

        raw_pvalues.append(p)
        rows.append({
            "hypothesis": hyp_name,
            "model_a": model_a,
            "model_b": model_b,
            "mean_error_a": errors_a.mean(),
            "mean_error_b": errors_b.mean(),
            "p_value_raw": p,
        })

    reject, corrected_pvalues, _, _ = multipletests(raw_pvalues, alpha=alpha, method="holm")

    for i, row in enumerate(rows):
        row["p_value_corrected"] = corrected_pvalues[i]
        row["signifikan"] = "Ya" if reject[i] else "Tidak"

    results_df = pd.DataFrame(rows)
    results_df.to_csv(config.SIGNIFICANCE_TEST_FILE, index=False)
    print(f"\n💾 Hasil uji signifikansi (4 hipotesis, Holm-Bonferroni) -> {config.SIGNIFICANCE_TEST_FILE}")

    return results_df


# ==========================================================
# 4. EFFECT SIZE + CI 95% (BOOTSTRAP) — PELENGKAP P-VALUE
# ==========================================================
def bootstrap_effect_size(aggregated_errors, model_a, model_b, n_boot=2000, seed=42):
    """
    Selisih mean error (model_a - model_b) + interval kepercayaan 95% lewat
    bootstrap resampling pada data uji -- dilaporkan meski hasil tidak
    signifikan, supaya besaran efek tetap terlihat, bukan diabaikan begitu
    saja (Subbab 3.9.2).
    """
    rng = np.random.default_rng(seed)
    errors_a = aggregated_errors[model_a]
    errors_b = aggregated_errors[model_b]
    n = len(errors_a)

    diffs = errors_a - errors_b
    observed_diff = diffs.mean()

    boot_diffs = np.array([
        rng.choice(diffs, size=n, replace=True).mean() for _ in range(n_boot)
    ])
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])

    return {
        "model_a": model_a,
        "model_b": model_b,
        "mean_diff": observed_diff,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
    }


def run_all_effect_sizes(aggregated_errors):
    rows = [
        bootstrap_effect_size(aggregated_errors, model_a, model_b)
        for _, model_a, model_b in PRE_REGISTERED_HYPOTHESES
    ]
    return pd.DataFrame(rows)