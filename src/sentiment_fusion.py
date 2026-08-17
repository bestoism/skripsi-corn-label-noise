import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

from src import config

SENTIMENT_CACHE_FILE = os.path.join(config.PROXY_CACHE_DIR, "sentiment_scores_w11wo.npy")
SENTIMENT_CACHE_META_FILE = os.path.join(config.PROXY_CACHE_DIR, "sentiment_scores_w11wo_meta.csv")


def compute_sentiment_scores(texts, batch_size=32):
    """
    Skor softmax dari model sentimen eksternal (w11wo/indonesian-roberta-base-
    sentiment-classifier) untuk tiap teks -- DIPAKAI SEBAGAI FITUR TAMBAHAN
    (bukan di-fine-tune). Model ini dibekukan (frozen) karena perannya di sini
    cuma menyuntikkan sinyal sentimen independen ke representasi IndoBERT,
    bukan ikut dilatih ulang -- kalau ikut di-fine-tune, sinyalnya akan
    bercampur dengan apa yang sudah dipelajari IndoBERT dan kita kehilangan
    poin fusinya (menggabungkan DUA sumber sinyal yang independen).

    Di-cache karena modelnya beda tokenizer dari IndoBERT -- precompute
    sekali, dipakai berulang di setiap fold K-Fold tanpa recompute.
    """
    if os.path.exists(SENTIMENT_CACHE_FILE) and os.path.exists(SENTIMENT_CACHE_META_FILE):
        meta = pd.read_csv(SENTIMENT_CACHE_META_FILE)
        if meta["text"].tolist() == list(texts):
            print("⚡ Memuat skor sentimen dari cache...")
            return np.load(SENTIMENT_CACHE_FILE)
        print("⚠️ Cache skor sentimen tidak cocok dgn data saat ini -> recompute.")

    print(f"⬇️  Memuat model sentimen: {config.SENTIMENT_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.SENTIMENT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(config.SENTIMENT_MODEL_NAME).to(config.DEVICE)
    model.eval()

    n_labels = model.config.num_labels
    print(f"   Jumlah kelas sentimen model ini: {n_labels} ({model.config.id2label})")

    scores = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment scoring"):
            batch_texts = texts[i:i + batch_size]
            encoded = tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=config.MAX_LEN, return_tensors="pt",
            ).to(config.DEVICE)
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            scores.append(probs)

    scores = np.concatenate(scores, axis=0)
    np.save(SENTIMENT_CACHE_FILE, scores)
    pd.DataFrame({"text": texts}).to_csv(SENTIMENT_CACHE_META_FILE, index=False)

    del model
    torch.cuda.empty_cache()
    return scores


def get_sentiment_dim():
    """Jumlah kelas model sentimen (biasanya 3: positive/neutral/negative) --
    dipakai models.py untuk menentukan ukuran input layer fusi."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(config.SENTIMENT_MODEL_NAME)
    return cfg.num_labels