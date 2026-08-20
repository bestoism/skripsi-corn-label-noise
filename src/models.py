"""
models.py -- Arsitektur model untuk baseline (Cross-Entropy), CORN (ordinal
regression), dan CORN+fusi sentimen eksternal (P5). Backbone tunggal
(config.PRETRAINED_MODEL_NAME = IndoBERT), sesuai Batasan Masalah Bab 1.6.
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from src import config


class IndoBERTStandard(nn.Module):
    """Model baseline: Cross-Entropy, output 5 neuron (satu neuron per kelas rating)."""

    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config.PRETRAINED_MODEL_NAME)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, config.NUM_CLASSES)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)
        return self.classifier(pooled)


class IndoBERTCORN(nn.Module):
    """Model ordinal regression (CORN): output K-1 neuron (4 neuron untuk 5 kelas rating)."""

    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config.PRETRAINED_MODEL_NAME)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, config.NUM_CLASSES - 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)
        return self.classifier(pooled)


class IndoBERTCORNFusion(nn.Module):
    """
    CORN + fusi skor sentimen eksternal (P5, Bab 3 Tabel 3.2).

    PERBAIKAN dari versi sebelumnya: skor sentimen (biasanya 3-dim:
    positive/neutral/negative) TIDAK lagi di-concat langsung ke representasi
    BERT (768-dim). Concat langsung berisiko membuat sinyal sentimen
    "tenggelam" secara numerik -- classifier linear di lapisan akhir bisa
    dengan mudah mengabaikan 3 dimensi kecil di antara 768 dimensi besar,
    bukan karena sinyalnya tidak berguna, tapi karena skalanya timpang.

    Solusi di sini: proyeksikan dulu skor sentimen ke ruang berdimensi lebih
    seimbang (sentiment_proj_dim, default 32) lewat satu linear layer +
    non-linearitas (GELU, konsisten dengan aktivasi internal BERT), BARU
    di-concat dengan representasi BERT. Ini memberi sentimen "ruang" yang
    proporsional untuk berkontribusi ke keputusan akhir, sekaligus tetap
    sederhana untuk dijelaskan/didebug dibanding gating mechanism.

    sentiment_dim = jumlah kelas model sentimen eksternal (biasanya 3).
    sentiment_proj_dim = dimensi hasil proyeksi (dapat dituning; 32 adalah
    titik awal yang wajar -- jauh lebih besar dari 3 tapi masih kecil
    dibanding 768, supaya BERT tetap dominan sebagai representasi utama).
    """

    def __init__(self, sentiment_dim, sentiment_proj_dim=32):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config.PRETRAINED_MODEL_NAME)
        self.dropout = nn.Dropout(0.3)

        self.sentiment_proj = nn.Sequential(
            nn.Linear(sentiment_dim, sentiment_proj_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        fused_dim = self.bert.config.hidden_size + sentiment_proj_dim
        self.classifier = nn.Linear(fused_dim, config.NUM_CLASSES - 1)

    def forward(self, input_ids, attention_mask, sentiment):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)

        sentiment_proj = self.sentiment_proj(sentiment)

        fused = torch.cat([pooled, sentiment_proj], dim=1)
        return self.classifier(fused)


def build_model(loss_type):
    """
    Factory kecil supaya train.py dan proxy.py tidak perlu tahu detail
    kelas mana yang dipanggil -- cukup sebut 'ce' atau 'corn'.
    """
    if loss_type == "ce":
        return IndoBERTStandard().to(config.DEVICE)
    elif loss_type == "corn":
        return IndoBERTCORN().to(config.DEVICE)
    else:
        raise ValueError(f"loss_type harus 'ce' atau 'corn', dapat: {loss_type}")