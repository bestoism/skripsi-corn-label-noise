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

class IndoBERTCORNFusion(nn.Module):
    """
    CORN, tapi representasi IndoBERT digabung (concat) dengan skor sentimen
    eksternal SEBELUM masuk classifier head -- ini yang jadi P5 di Bab 3.
    sentiment_dim = jumlah kelas model sentimen eksternal (biasanya 3).
    """

    def __init__(self, sentiment_dim):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config.PRETRAINED_MODEL_NAME)
        self.dropout = nn.Dropout(0.3)
        fused_dim = self.bert.config.hidden_size + sentiment_dim
        self.classifier = nn.Linear(fused_dim, config.NUM_CLASSES - 1)

    def forward(self, input_ids, attention_mask, sentiment):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)
        fused = torch.cat([pooled, sentiment], dim=1)
        return self.classifier(fused)