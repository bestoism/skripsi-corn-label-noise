import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim import AdamW
from coral_pytorch.losses import corn_loss
from coral_pytorch.dataset import corn_label_from_logits

from src import config
from src.data import ReviewDataset
from src.models import build_model
from src.metrics import compute_metrics

torch.backends.cudnn.benchmark = True


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _evaluate(model, loader, loss_type):
    """Dipakai untuk validation tiap epoch, dan test set sekali di akhir."""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(config.DEVICE)
            attention_mask = batch["attention_mask"].to(config.DEVICE)
            lbl = batch["labels"].numpy()

            logits = model(input_ids, attention_mask)
            p = (torch.argmax(logits, dim=1) if loss_type == "ce"
                 else corn_label_from_logits(logits)).cpu().numpy()

            preds.extend(p)
            labels.extend(lbl)
    return compute_metrics(labels, preds)


def run_experiment(scenario_name, train_path, loss_type, seed):
    """
    Melatih satu skenario (mis. M4_Baseline_CORN) dengan satu seed.
    Test set dievaluasi HANYA SEKALI di akhir, pakai checkpoint terbaik
    berdasarkan validation MAE (bukan test set -- test set tidak boleh
    dipakai untuk memilih model, itu akan jadi data leakage).
    """
    print(f"\n🚀 {scenario_name} | Seed: {seed} | Loss: {loss_type.upper()}")
    set_seed(seed)

    df_train_full = pd.read_csv(train_path)

    if config.DEBUG_MODE and len(df_train_full) > config.DEBUG_SAMPLE_SIZE:
        df_train_full, _ = train_test_split(
            df_train_full, train_size=config.DEBUG_SAMPLE_SIZE,
            random_state=seed, stratify=df_train_full["rating"],
        )
        print(f"   [DEBUG] subset -> {len(df_train_full)} baris")

    # split train/val dari data skenario ini; test set TIDAK disentuh di sini
    df_train, df_val = train_test_split(
        df_train_full, test_size=config.VAL_SIZE,
        random_state=seed, stratify=df_train_full["rating"],
    )
    df_test = pd.read_csv(config.TEST_FILE)

    train_loader = DataLoader(
        ReviewDataset(df_train["cleaned_text"].tolist(), df_train["rating"].tolist()),
        batch_size=config.BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        ReviewDataset(df_val["cleaned_text"].tolist(), df_val["rating"].tolist()),
        batch_size=config.BATCH_SIZE, shuffle=False,
    )
    test_loader = DataLoader(
        ReviewDataset(df_test["cleaned_text"].tolist(), df_test["rating"].tolist()),
        batch_size=config.BATCH_SIZE, shuffle=False,
    )

    model = build_model(loss_type)
    criterion = nn.CrossEntropyLoss() if loss_type == "ce" else None
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    best_val_mae = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(config.DEVICE)
            attention_mask = batch["attention_mask"].to(config.DEVICE)
            labels = batch["labels"].to(config.DEVICE)

            with torch.cuda.amp.autocast():
                logits = model(input_ids, attention_mask)
                loss = (criterion(logits, labels) if loss_type == "ce"
                        else corn_loss(logits, labels, num_classes=config.NUM_CLASSES))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        val_metrics = _evaluate(model, val_loader, loss_type)
        val_mae = val_metrics["mae"]

        print(f"Epoch {epoch + 1}/{config.EPOCHS} - Loss: {train_loss / len(train_loader):.4f} "
              f"- Val MAE: {val_mae:.4f} | QWK: {val_metrics['qwk']:.4f} "
              f"| Off-by-1: {val_metrics['off_by_one']:.4f} | Acc: {val_metrics['accuracy']:.4f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"   ⏹ Early stopping di epoch {epoch + 1} (Val MAE tidak membaik {config.PATIENCE}x)")
                break

    model.load_state_dict(best_state)

    ckpt_path = os.path.join(config.MODEL_CKPT_DIR, scenario_name, f"seed{seed}_best.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(best_state, ckpt_path)

    test_metrics = _evaluate(model, test_loader, loss_type)
    print(f"🏆 Test (dari model Val MAE terbaik={best_val_mae:.4f}): "
          f"MAE={test_metrics['mae']:.4f} | RMSE={test_metrics['rmse']:.4f} | "
          f"Acc={test_metrics['accuracy']:.4f} | Off-by-1={test_metrics['off_by_one']:.4f} | "
          f"QWK={test_metrics['qwk']:.4f}")

    return test_metrics