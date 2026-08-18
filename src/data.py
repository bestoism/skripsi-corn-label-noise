import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from src import config


class ReviewDataset(Dataset):
    """
    Dataset PyTorch untuk teks ulasan + rating.
    Menerima label RAW 1-5 (bukan 0-4) supaya pemanggilnya (train.py, proxy.py)
    tidak perlu ingat konversi index -- semua dilakukan di sini, satu tempat.
    """

    _tokenizer_cache = {}  # FIX: dict keyed by model name, bukan satu slot global

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = [int(l) - 1 for l in labels]  # 1-5 -> 0-4 (syarat PyTorch & CORN)

        model_name = config.PRETRAINED_MODEL_NAME
        if model_name not in ReviewDataset._tokenizer_cache:
            ReviewDataset._tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer = ReviewDataset._tokenizer_cache[model_name]

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

        model_name = config.PRETRAINED_MODEL_NAME
        if model_name not in ReviewDataset._tokenizer_cache:
            ReviewDataset._tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer = ReviewDataset._tokenizer_cache[model_name]

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