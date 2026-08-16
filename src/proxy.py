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
def _get_embeddings(texts, pooling="mean", batch_size=32):
    """pooling: 'cls' atau 'mean'. Cache mengikuti nama pooling agar tidak tercampur."""
    cache_file = config.EMBEDDING_CACHE_FILE.replace(".npy", f"_{pooling}.npy")
    meta_file = config.EMBEDDING_CACHE_META_FILE.replace(".csv", f"_{pooling}.csv")

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
    X = _get_embeddings(texts, pooling=pooling, batch_size=config.BATCH_SIZE)
    base_clf = LogisticRegression(max_iter=2000, random_state=42)
    calibrated = CalibratedClassifierCV(base_clf, cv=3, method="sigmoid")
    return cross_val_predict(
        calibrated, X, labels, cv=config.PROXY_CV_FOLDS, method="predict_proba", n_jobs=-1
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
# FINE-TUNE K-FOLD GENERIK (dipakai proxy 2, 3, 4)
# loss_type: "ce" atau "corn" | extra_input: None atau tensor sentimen (proxy 4)
# ==========================================================
def _finetune_kfold_oof(texts, labels, loss_type, use_sentiment_fusion=False):
    cached = _load_cache_if_valid(config.PROXY_PRED_PROBS_FILE, config.PROXY_PRED_PROBS_META_FILE, texts)
    if cached is not None:
        return cached

    torch.manual_seed(42)
    n = len(texts)
    oof = np.zeros((n, config.NUM_CLASSES), dtype=np.float32)
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
        fold_probs = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(config.DEVICE)
                attention_mask = batch["attention_mask"].to(config.DEVICE)
                logits = model(input_ids, attention_mask)
                probs = (F.softmax(logits, dim=1) if loss_type == "ce"
                         else _corn_logits_to_probas(logits)).cpu().numpy()
                fold_probs.append(probs)

        oof[val_idx] = np.concatenate(fold_probs, axis=0)
        del model, optimizer
        torch.cuda.empty_cache()

    _save_cache(config.PROXY_PRED_PROBS_FILE, config.PROXY_PRED_PROBS_META_FILE, texts, oof)
    return oof


# ==========================================================
# DISPATCHER — SATU-SATUNYA FUNGSI YANG DIPANGGIL DARI clean.py
# ==========================================================
def get_proxy_pred_probs(texts, labels):
    """
    Mengembalikan OOF pred_probs sesuai config.PROXY_ID (0-4).
    Ganti proxy cukup ubah config.PROXY_ID -- tidak perlu sentuh file ini.
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
        # NOTE: fusi sentimen eksternal (P5) butuh implementasi tambahan di
        # model forward (concat skor sentimen sebelum classifier head).
        # Ditandai belum diimplementasikan penuh di sini -- lihat catatan di bawah.
        raise NotImplementedError(
            "Proxy 4 (fusion) butuh varian model IndoBERTCORNFusion terpisah "
            "di models.py. Beri tahu saya kalau kamu mau lanjut ke proxy ini "
            "-- kita tambahkan sebagai langkah berikutnya."
        )
    else:
        raise ValueError(f"PROXY_ID tidak dikenal: {config.PROXY_ID} (harus 0-4)")