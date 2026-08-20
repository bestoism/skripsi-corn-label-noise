"""
proxy.py -- Implementasi semua proxy classifier (P1-P5) untuk Confident
Learning. Backbone tunggal (config.PRETRAINED_MODEL_NAME = IndoBERT),
sesuai Batasan Masalah Bab 1.6 -- TIDAK ADA proxy dengan backbone lain
(P6/IndoBERTweet dihapus total, lihat config.py).
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from tqdm import tqdm

from src import config
from src.data import ReviewDataset
from src.models import IndoBERTStandard, IndoBERTCORN
from coral_pytorch.losses import corn_loss

from src.sentiment_fusion import compute_sentiment_scores, get_sentiment_dim
from src.data import ReviewDatasetFusion
from src.models import IndoBERTCORNFusion


# ==========================================================
# CACHE HELPER — dipakai semua metode, supaya tidak recompute
# ==========================================================
def _load_cache_if_valid(cache_file, meta_file, texts):
    if os.path.exists(cache_file) and os.path.exists(meta_file):
        meta = pd.read_csv(meta_file)
        if meta["text"].tolist() == list(texts):
            print(f"⚡ Memuat cache proxy [{config.PROXY_NAME}] ...")
            return np.load(cache_file)
        print(f"⚠️ Cache proxy [{config.PROXY_NAME}] tidak cocok dgn data saat ini -> recompute.")
    return None


def _save_cache(cache_file, meta_file, texts, array):
    np.save(cache_file, array)
    pd.DataFrame({"text": texts}).to_csv(meta_file, index=False)


# ==========================================================
# EMBEDDING BEKU (dipakai oleh proxy 0 & 1)
# ==========================================================
# CATATAN METODOLOGIS: tokenisasi di sini (padding=True, dinamis per-batch)
# BEDA strategi dengan ReviewDataset yang dipakai P3/P4/P5 (padding="max_length",
# fixed 128 token). Ini bukan kesalahan -- keduanya valid secara teknis --
# tapi berarti P1/P2 vs P3/P4/P5 diproses lewat jalur tokenisasi yang tidak
# identik. Sebutkan ini di Bab III/IV sebagai potential confound minor kalau
# nanti menganalisis kenapa performa P1/P2 jauh di bawah P3/P4/P5 -- selisih
# performanya kemungkinan besar didominasi oleh fine-tuning vs frozen
# embedding, tapi strategi padding yang beda tetap perlu dicatat sebagai
# variabel yang tidak sepenuhnya dikontrol.
def _get_embeddings(texts, pooling="mean", batch_size=32):
    """pooling: 'cls' atau 'mean'. Cache mengikuti nama pooling agar tidak tercampur."""
    cache_file = config.EMBEDDING_CACHE_FILE_BASE.replace(".npy", f"_{pooling}.npy")
    meta_file = config.EMBEDDING_CACHE_META_FILE_BASE.replace(".csv", f"_{pooling}.csv")

    cached = _load_cache_if_valid(cache_file, meta_file, texts)
    if cached is not None:
        return cached

    tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_MODEL_NAME)
    model = AutoModel.from_pretrained(config.PRETRAINED_MODEL_NAME).to(config.DEVICE)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding ({pooling})"):
            batch_texts = texts[i: i + batch_size]
            encoded = tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=config.MAX_LEN, return_tensors="pt",
            ).to(config.DEVICE)
            outputs = model(**encoded)

            if pooling == "cls":
                emb = outputs.last_hidden_state[:, 0, :]
            else:  # mean pooling
                token_emb = outputs.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).expand(token_emb.size()).float()
                emb = torch.sum(token_emb * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)

            embeddings.extend(emb.cpu().numpy())

    embeddings = np.array(embeddings)
    _save_cache(cache_file, meta_file, texts, embeddings)
    return embeddings


# ==========================================================
# PROXY 0 & 1: embedding beku + Logistic Regression
# ==========================================================
def _proxy_frozen_lr(texts, labels, pooling):
    """
    PERBAIKAN: cv sebelumnya diberikan sebagai integer (config.PROXY_CV_FOLDS),
    yang membuat cross_val_predict() default ke StratifiedKFold(shuffle=False)
    -- karena data kemungkinan terurut per-app dan per-waktu scraping (lihat
    scrape_google_play.py), fold yang dihasilkan tanpa shuffle berisiko bias
    secara app/waktu. Sekarang StratifiedKFold dibuat eksplisit dengan
    shuffle=True, random_state=42 -- IDENTIK dengan skema CV yang dipakai
    _finetune_kfold_oof (P3/P4/P5), supaya perbandingan lintas proxy di
    pilot study benar-benar apple-to-apple dari sisi skema validasi silang.
    """
    X = _get_embeddings(texts, pooling=pooling, batch_size=config.BATCH_SIZE)
    base_clf = LogisticRegression(max_iter=2000, random_state=42)
    calibrated = CalibratedClassifierCV(base_clf, cv=3, method="sigmoid")

    cv_splitter = StratifiedKFold(
        n_splits=config.PROXY_CV_FOLDS, shuffle=True, random_state=42
    )
    return cross_val_predict(
        calibrated, X, labels, cv=cv_splitter, method="predict_proba", n_jobs=-1
    )


# ==========================================================
# KONVERSI LOGIT CORN -> PROBABILITAS PENUH (dipakai proxy 3 & 4)
# ==========================================================
def _corn_logits_to_probas(logits):
    """P(y>k) = cumprod sigmoid -> otomatis monoton turun (rank-consistent)."""
    probs_cond = torch.sigmoid(logits)
    cum_probs = torch.cumprod(probs_cond, dim=1)

    batch_size, K = logits.shape[0], logits.shape[1] + 1
    class_probs = torch.zeros(batch_size, K, device=logits.device, dtype=logits.dtype)
    class_probs[:, 0] = 1.0 - cum_probs[:, 0]
    for k in range(1, K - 1):
        class_probs[:, k] = cum_probs[:, k - 1] - cum_probs[:, k]
    class_probs[:, K - 1] = cum_probs[:, K - 2]

    class_probs = torch.clamp(class_probs, min=1e-8)
    return class_probs / class_probs.sum(dim=1, keepdim=True)


# ==========================================================
# TEMPERATURE SCALING PASCA-FOLD (BARU) — dipakai P3/P4/P5
# ==========================================================
def _calibrate_with_temperature(oof_logits, labels_arr, loss_type):
    """
    P1/P2 sudah dikalibrasi lewat CalibratedClassifierCV (Platt scaling),
    tapi P3/P4/P5 (fine-tuned) sebelumnya TIDAK dikalibrasi sama sekali --
    softmax/CORN-probas mentah langsung dipakai sebagai pred_probs untuk
    Confident Learning, padahal cleanlab (Northcutt dkk., 2021) eksplisit
    mengasumsikan pred_probs yang sudah well-calibrated untuk estimasi
    confident joint yang akurat.

    Solusi di sini: temperature scaling (Guo dkk., 2017) -- satu skalar T
    dicari lewat LBFGS untuk meminimalkan negative log-likelihood terhadap
    label ASLI (bukan label mayoritas atau proxy lain), dihitung dari OOF
    logits GABUNGAN seluruh fold (bukan per-fold, supaya estimasi T lebih
    stabil dan tidak overfit ke satu fold kecil).

    Untuk CE: probs_calibrated = softmax(logits / T)
    Untuk CORN: probs_calibrated = _corn_logits_to_probas(logits / T)
    -- generalisasi temperature scaling standar (yang aslinya didefinisikan
    untuk softmax) ke struktur cumulative-link CORN, dengan membagi logit
    MENTAH (sebelum sigmoid) dengan T. Ini pilihan desain yang masuk akal
    (T besar -> distribusi makin rata/kurang percaya diri, T kecil -> makin
    tajam, konsisten dengan interpretasi T pada softmax), tapi bukan
    turunan formal dari teori kalibrasi CORN yang sudah divalidasi di
    literatur -- sebutkan ini eksplisit sebagai keputusan metodologis kalau
    ditanya penguji, jangan diklaim sebagai "standar baku".

    T awal = 1.0 (tidak ada scaling) -- kalau optimasi gagal konvergen,
    fallback ke T=1.0 (probs tidak berubah) dengan peringatan di log.
    """
    logits_t = torch.tensor(oof_logits, dtype=torch.float32)
    labels_t = torch.tensor(labels_arr, dtype=torch.long)

    temperature = torch.nn.Parameter(torch.ones(1) * 1.0)
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=100)

    def _nll_loss(T):
        T_clamped = torch.clamp(T, min=1e-2)  # cegah pembagian oleh nol/negatif
        scaled_logits = logits_t / T_clamped
        if loss_type == "ce":
            log_probs = F.log_softmax(scaled_logits, dim=1)
        else:  # corn
            probs = _corn_logits_to_probas(scaled_logits)
            log_probs = torch.log(torch.clamp(probs, min=1e-8))
        return F.nll_loss(log_probs, labels_t)

    def closure():
        optimizer.zero_grad()
        loss = _nll_loss(temperature)
        loss.backward()
        return loss

    try:
        nll_before = _nll_loss(temperature).item()
        optimizer.step(closure)
        nll_after = _nll_loss(temperature).item()
        T_final = torch.clamp(temperature.detach(), min=1e-2).item()
        print(f"   🌡️  Temperature scaling [{config.PROXY_NAME}]: T={T_final:.4f} "
              f"(NLL {nll_before:.4f} -> {nll_after:.4f})")
    except Exception as e:
        print(f"   ⚠️ Temperature scaling gagal konvergen ({e}) -- fallback T=1.0 (tanpa scaling).")
        T_final = 1.0

    with torch.no_grad():
        scaled_logits = logits_t / T_final
        if loss_type == "ce":
            probs_calibrated = F.softmax(scaled_logits, dim=1)
        else:
            probs_calibrated = _corn_logits_to_probas(scaled_logits)

    return probs_calibrated.numpy(), T_final


# ==========================================================
# FINE-TUNE K-FOLD GENERIK (dipakai proxy 2, 3) — dengan temperature scaling
# ==========================================================
# CATATAN METODOLOGIS: setiap fold dilatih fixed 3 epoch
# (config.PROXY_FINETUNE_EPOCHS) TANPA validasi/early-stopping di dalam
# fold -- ini keputusan desain yang disengaja untuk keperluan OOF generation
# (beda dari train.py yang MEMANG pakai early-stopping untuk model final),
# tapi berarti kualitas pred_probs per fold bisa under/overfit tanpa kontrol
# eksplisit. Sebutkan ini sebagai batasan metodologis di Bab III/IV --
# temperature scaling di atas MEMBANTU mengoreksi overconfidence akibat hal
# ini, tapi tidak sepenuhnya menggantikan kontrol early-stopping per fold.
def _finetune_kfold_oof(texts, labels, loss_type):
    cached = _load_cache_if_valid(config.PROXY_PRED_PROBS_FILE, config.PROXY_PRED_PROBS_META_FILE, texts)
    if cached is not None:
        return cached

    torch.manual_seed(42)
    n = len(texts)
    n_raw_outputs = config.NUM_CLASSES if loss_type == "ce" else config.NUM_CLASSES - 1
    oof_logits = np.zeros((n, n_raw_outputs), dtype=np.float32)
    texts_arr = np.array(texts, dtype=object)
    labels_arr = np.array(labels)  # 0-indexed

    skf = StratifiedKFold(n_splits=config.PROXY_CV_FOLDS, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts_arr, labels_arr)):
        print(f"   [Proxy {config.PROXY_NAME}] Fold {fold + 1}/{config.PROXY_CV_FOLDS} "
              f"(train={len(train_idx)}, val={len(val_idx)})...")

        model = (IndoBERTCORN() if loss_type == "corn" else IndoBERTStandard()).to(config.DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.PROXY_FINETUNE_LR)
        criterion = nn.CrossEntropyLoss() if loss_type == "ce" else None

        train_ds = ReviewDataset(texts_arr[train_idx].tolist(), (labels_arr[train_idx] + 1).tolist())
        val_ds = ReviewDataset(texts_arr[val_idx].tolist(), (labels_arr[val_idx] + 1).tolist())
        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

        scaler = torch.cuda.amp.GradScaler()
        model.train()
        for epoch in range(config.PROXY_FINETUNE_EPOCHS):
            epoch_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(config.DEVICE)
                attention_mask = batch["attention_mask"].to(config.DEVICE)
                lbl = batch["labels"].to(config.DEVICE)

                with torch.cuda.amp.autocast():
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, lbl) if loss_type == "ce" else corn_loss(logits, lbl, num_classes=config.NUM_CLASSES)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
            print(f"      Epoch {epoch + 1}/{config.PROXY_FINETUNE_EPOCHS} - Loss: {epoch_loss / len(train_loader):.4f}")

        model.eval()
        fold_logits = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(config.DEVICE)
                attention_mask = batch["attention_mask"].to(config.DEVICE)
                logits = model(input_ids, attention_mask)
                fold_logits.append(logits.cpu().numpy())

        oof_logits[val_idx] = np.concatenate(fold_logits, axis=0)
        del model, optimizer
        torch.cuda.empty_cache()

    # Kalibrasi: T dicari dari OOF logits gabungan seluruh fold, BUKAN
    # per-fold -- lihat docstring _calibrate_with_temperature().
    oof_probs_calibrated, _T = _calibrate_with_temperature(oof_logits, labels_arr, loss_type)

    _save_cache(config.PROXY_PRED_PROBS_FILE, config.PROXY_PRED_PROBS_META_FILE, texts, oof_probs_calibrated)
    return oof_probs_calibrated


# ==========================================================
# FINE-TUNE FUSION K-FOLD (dipakai proxy 4) — dengan temperature scaling
# ==========================================================
def _finetune_fusion_kfold_oof(texts, labels):
    """P5: sama seperti _finetune_kfold_oof (CORN), tapi model & dataset-nya
    versi fusion -- representasi IndoBERT digabung skor sentimen eksternal
    (sudah diproyeksikan, lihat models.py) sebelum classifier CORN. Sama
    seperti P3/P4, sekarang dikalibrasi lewat temperature scaling pasca-fold."""
    cached = _load_cache_if_valid(config.PROXY_PRED_PROBS_FILE, config.PROXY_PRED_PROBS_META_FILE, texts)
    if cached is not None:
        return cached

    sentiment_scores = compute_sentiment_scores(texts)
    sentiment_dim = get_sentiment_dim()

    torch.manual_seed(42)
    n = len(texts)
    oof_logits = np.zeros((n, config.NUM_CLASSES - 1), dtype=np.float32)
    texts_arr = np.array(texts, dtype=object)
    labels_arr = np.array(labels)

    skf = StratifiedKFold(n_splits=config.PROXY_CV_FOLDS, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts_arr, labels_arr)):
        print(f"   [Proxy {config.PROXY_NAME}] Fold {fold + 1}/{config.PROXY_CV_FOLDS} "
              f"(train={len(train_idx)}, val={len(val_idx)})...")

        model = IndoBERTCORNFusion(sentiment_dim).to(config.DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.PROXY_FINETUNE_LR)

        train_ds = ReviewDatasetFusion(
            texts_arr[train_idx].tolist(), (labels_arr[train_idx] + 1).tolist(),
            sentiment_scores[train_idx],
        )
        val_ds = ReviewDatasetFusion(
            texts_arr[val_idx].tolist(), (labels_arr[val_idx] + 1).tolist(),
            sentiment_scores[val_idx],
        )
        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

        scaler = torch.cuda.amp.GradScaler()
        model.train()
        for epoch in range(config.PROXY_FINETUNE_EPOCHS):
            epoch_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(config.DEVICE)
                attention_mask = batch["attention_mask"].to(config.DEVICE)
                sentiment = batch["sentiment"].to(config.DEVICE)
                lbl = batch["labels"].to(config.DEVICE)

                with torch.cuda.amp.autocast():
                    logits = model(input_ids, attention_mask, sentiment)
                    loss = corn_loss(logits, lbl, num_classes=config.NUM_CLASSES)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
            print(f"      Epoch {epoch + 1}/{config.PROXY_FINETUNE_EPOCHS} - Loss: {epoch_loss / len(train_loader):.4f}")

        model.eval()
        fold_logits = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(config.DEVICE)
                attention_mask = batch["attention_mask"].to(config.DEVICE)
                sentiment = batch["sentiment"].to(config.DEVICE)
                logits = model(input_ids, attention_mask, sentiment)
                fold_logits.append(logits.cpu().numpy())

        oof_logits[val_idx] = np.concatenate(fold_logits, axis=0)
        del model, optimizer
        torch.cuda.empty_cache()

    oof_probs_calibrated, _T = _calibrate_with_temperature(oof_logits, labels_arr, loss_type="corn")

    _save_cache(config.PROXY_PRED_PROBS_FILE, config.PROXY_PRED_PROBS_META_FILE, texts, oof_probs_calibrated)
    return oof_probs_calibrated


# ==========================================================
# DISPATCHER — SATU-SATUNYA FUNGSI YANG DIPANGGIL DARI clean.py
# ==========================================================
def get_proxy_pred_probs(texts, labels):
    """
    Mengembalikan OOF pred_probs (SUDAH DIKALIBRASI untuk P2-P4) sesuai
    config.PROXY_ID (0-4). Ganti proxy cukup ubah config.PROXY_ID -- tidak
    perlu sentuh file ini. HANYA 5 PROXY (P1-P5, id 0-4), sesuai Batasan
    Masalah Bab 1.6 -- tidak ada proxy dengan backbone di luar IndoBERT.
    """
    print(f"\n🧮 Menghitung OOF pred_probs — proxy [{config.PROXY_ID}] {config.PROXY_NAME}")

    if config.PROXY_ID == 0:
        return _proxy_frozen_lr(texts, labels, pooling="cls")
    elif config.PROXY_ID == 1:
        return _proxy_frozen_lr(texts, labels, pooling="mean")
    elif config.PROXY_ID == 2:
        return _finetune_kfold_oof(texts, labels, loss_type="ce")
    elif config.PROXY_ID == 3:
        return _finetune_kfold_oof(texts, labels, loss_type="corn")
    elif config.PROXY_ID == 4:
        return _finetune_fusion_kfold_oof(texts, labels)
    else:
        raise ValueError(
            f"PROXY_ID tidak dikenal: {config.PROXY_ID} (harus 0-4, sesuai 5 tahap "
            f"ablasi Tabel 3.2 -- backbone di luar IndoBERT di luar cakupan penelitian, "
            f"lihat Batasan Masalah Bab 1.6)."
        )