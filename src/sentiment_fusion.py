"""
sentiment_fusion.py -- Skor sentimen eksternal (frozen) untuk P5 (fusi CORN).

MODEL YANG DIPAKAI: w11wo/indonesian-roberta-base-sentiment-classifier,
fine-tuned RoBERTa Indonesia di atas dataset SmSA (IndoNLU, Wilie dkk. 2020
-- sudah dikutip di Tinjauan Pustaka [15]). Evaluasi resmi model ini:
akurasi 93,2%, F1-macro 91,02% di test set benchmark publik SmSA.

LIMITASI YANG WAJIB DISEBUTKAN DI BAB III/IV (bukan bug, keputusan desain
yang perlu dijustifikasi): SmSA berisi komentar & ulasan Indonesia secara
umum, BUKAN korpus yang spesifik ulasan aplikasi/e-commerce seperti data
penelitian ini (Tokopedia/Gojek/SeaBank). Ada domain gap antara data latih
model sentimen ini dengan data ulasan app store yang dianalisis skripsi ini
-- kalau P5 (fusion) tidak memberi manfaat dibanding P4 (CORN murni) di
Bab IV, domain mismatch ini adalah salah satu penjelasan yang masuk akal
dan harus disebutkan, bukan disembunyikan.

Alternatif yang dipertimbangkan tapi TIDAK dipakai (dicatat untuk
transparansi metodologis):
- w11wo/indonesian-roberta-base-prdect-id: dilatih di atas PRDECT-ID
  (Sutoyo dkk., 2022 -- dataset yang sama dengan ref [19] di Tinjauan
  Pustaka), domain e-commerce jauh lebih dekat. TIDAK dipakai karena
  labelnya 5 EMOSI (Happy/Anger/Sadness/Love/Fear, teori Shaver), bukan
  polaritas sentimen (positif/netral/negatif) -- mengganti ke model ini
  berarti mengubah desain P5 dari "fusi sentimen" jadi "fusi emosi", yang
  perlu direfleksikan ulang di Bab 3 (Tabel 3.2), bukan sekadar ganti nama
  model.
- agufsamudra/indo-sentiment-analysis: dilatih langsung di atas ulasan
  Play Store (domain paling dekat dari semua opsi yang ditemukan), TAPI
  tidak ada metrik evaluasi resmi yang dipublikasikan dan kredibilitasnya
  jauh di bawah w11wo untuk dikutip di skripsi -- domain-fit yang lebih
  baik tidak sepadan dengan risiko mengutip model yang tidak terverifikasi.

Kalau suatu saat model ini ingin diganti, cukup ubah config.SENTIMENT_MODEL_NAME
-- kode di bawah generik terhadap jumlah kelas asal modelnya adalah model
klasifikasi teks tunggal (single-text sequence classification).
"""

import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

from src import config

SENTIMENT_CACHE_FILE = os.path.join(
    config.PROXY_CACHE_DIR,
    f"sentiment_scores__{config.SENTIMENT_MODEL_NAME.replace('/', '_')}.npy",
)
SENTIMENT_CACHE_META_FILE = os.path.join(
    config.PROXY_CACHE_DIR,
    f"sentiment_scores__{config.SENTIMENT_MODEL_NAME.replace('/', '_')}_meta.csv",
)

# Label yang diharapkan kalau model sentimen diganti -- bukan validasi ketat
# (model lain boleh punya nama label berbeda), hanya sanity-check untuk
# mendeteksi kalau ganti config.SENTIMENT_MODEL_NAME ke model dengan jumlah
# kelas/jenis label yang jauh berbeda (mis. keliru pasang model emosi 5-kelas
# di sini) -- supaya ketahuan di log, bukan diam-diam salah interpretasi.
_EXPECTED_LABEL_COUNT_RANGE = (2, 3)  # sentimen polaritas: biner atau pos/neu/neg


def compute_sentiment_scores(texts, batch_size=32):
    """
    Skor softmax dari model sentimen eksternal untuk tiap teks -- DIPAKAI
    SEBAGAI FITUR TAMBAHAN (bukan di-fine-tune). Model ini dibekukan
    (frozen) karena perannya di sini cuma menyuntikkan sinyal sentimen
    independen ke representasi IndoBERT, bukan ikut dilatih ulang -- kalau
    ikut di-fine-tune, sinyalnya akan bercampur dengan apa yang sudah
    dipelajari IndoBERT dan kita kehilangan poin fusinya (menggabungkan DUA
    sumber sinyal yang independen).

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

    if not (_EXPECTED_LABEL_COUNT_RANGE[0] <= n_labels <= _EXPECTED_LABEL_COUNT_RANGE[1]):
        print(f"   ⚠️  PERINGATAN: model ini punya {n_labels} kelas, di luar rentang yang "
              f"diharapkan untuk sentimen polaritas ({_EXPECTED_LABEL_COUNT_RANGE[0]}-"
              f"{_EXPECTED_LABEL_COUNT_RANGE[1]} kelas). Kalau ini disengaja (mis. sengaja "
              f"ganti ke model emosi multi-kelas), pastikan Bab 3 (Tabel 3.2, deskripsi P5) "
              f"diperbarui juga -- desain 'fusi sentimen' berubah maknanya kalau sinyal "
              f"eksternalnya bukan lagi polaritas sentimen.")

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
    dipakai models.py untuk menentukan ukuran input layer proyeksi fusi
    (lihat IndoBERTCORNFusion di models.py)."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(config.SENTIMENT_MODEL_NAME)
    return cfg.num_labels