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

    _tokenizer = None  # tokenizer di-cache di level class, tidak di-load ulang tiap instance

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = [int(l) - 1 for l in labels]  # 1-5 -> 0-4 (syarat PyTorch & CORN)

        if ReviewDataset._tokenizer is None:
            ReviewDataset._tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_MODEL_NAME)
        self.tokenizer = ReviewDataset._tokenizer

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