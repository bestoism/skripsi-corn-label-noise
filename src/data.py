"""
data.py -- PyTorch Dataset untuk teks ulasan + rating, dan varian dengan
skor sentimen eksternal (untuk P5/fusion). Backbone tunggal
(config.PRETRAINED_MODEL_NAME = IndoBERT), sesuai Batasan Masalah Bab 1.6
-- tokenizer di-cache sebagai satu instance modul, bukan dict per-model,
karena tidak ada skenario ganti backbone di proyek ini.
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from src import config

_tokenizer = None  # cache tunggal, di-load sekali saat dibutuhkan pertama kali


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_MODEL_NAME)
    return _tokenizer


class ReviewDataset(Dataset):
    """
    Dataset PyTorch untuk teks ulasan + rating.
    Menerima label RAW 1-5 (bukan 0-4) supaya pemanggilnya (train.py, proxy.py)
    tidak perlu ingat konversi index -- semua dilakukan di sini, satu tempat.
    """

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = [int(l) - 1 for l in labels]  # 1-5 -> 0-4 (syarat PyTorch & CORN)
        self.tokenizer = _get_tokenizer()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=config.MAX_LEN,
            padding="max_length",
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class ReviewDatasetFusion(Dataset):
    """
    Varian ReviewDataset yang juga membawa skor sentimen eksternal
    (precomputed) per baris -- dipakai khusus untuk IndoBERTCORNFusion (P5).
    sentiment_scores harus array numpy dengan urutan baris SAMA PERSIS
    dengan texts (tanggung jawab pemanggil menjaga urutan ini konsisten).
    """

    def __init__(self, texts, labels, sentiment_scores):
        self.texts = texts
        self.labels = [int(l) - 1 for l in labels]
        self.sentiment_scores = sentiment_scores
        self.tokenizer = _get_tokenizer()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text, add_special_tokens=True, max_length=config.MAX_LEN,
            padding="max_length", truncation=True,
            return_token_type_ids=False, return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "sentiment": torch.tensor(self.sentiment_scores[idx], dtype=torch.float),
            "labels": torch.tensor(label, dtype=torch.long),
        }