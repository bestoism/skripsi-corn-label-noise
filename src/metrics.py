import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    cohen_kappa_score,
)


def compute_metrics(true_labels, predictions):
    """
    Menghitung lima metrik evaluasi untuk task ordinal 5-kelas (rating 1-5).
    Dipakai di tiga tempat berbeda dengan makna berbeda (jangan dicampur saat
    dilaporkan -- lihat Subbab 3.9.1 proposal):
      1. Kualitas proxy classifier (di clean.py, terhadap label mentah/noisy)
      2. Performa model final (di train.py, terhadap test set bersih)
      3. (tidak dipakai langsung di sini, tapi agreement rate validasi manusia
         di human_validation.py memakai logika terpisah karena labelnya
         kategorikal noise/not_noise/ambiguous, bukan numerik)

    Menerima label & prediksi dalam skala apa pun asal konsisten (0-4 atau 1-5)
    -- MAE/RMSE/Off-by-1 tidak peduli offset selama kedua argumen konsisten,
    tapi QWK sedikit sensitif terhadap rentang skala (lihat Doewes dkk., 2023),
    jadi sebaiknya selalu panggil dengan skala yang sama di semua pemanggilan
    (proyek ini konsisten pakai label 0-4 di internal, 1-5 saat print/laporan).
    """
    true_labels = np.asarray(true_labels)
    predictions = np.asarray(predictions)

    if true_labels.shape != predictions.shape:
        raise ValueError(
            f"Ukuran true_labels ({true_labels.shape}) dan predictions "
            f"({predictions.shape}) tidak sama."
        )

    mae = mean_absolute_error(true_labels, predictions)
    rmse = np.sqrt(mean_squared_error(true_labels, predictions))
    acc = accuracy_score(true_labels, predictions)

    # Off-by-one: proporsi prediksi yang meleset maksimal 1 tingkat dari label asli.
    # Penting untuk task ordinal ambigu -- bedakan kesalahan "tipis" vs "jauh".
    off_by_one = float(np.mean(np.abs(true_labels - predictions) <= 1))

    # Quadratic Weighted Kappa: makin jauh selisih prediksi-label, makin besar
    # penalti (kuadratik) -- metrik standar untuk ordinal regression/klasifikasi.
    qwk = cohen_kappa_score(true_labels, predictions, weights="quadratic")

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "accuracy": float(acc),
        "off_by_one": off_by_one,
        "qwk": float(qwk),
    }


def format_metrics_row(metrics, precision=4):
    """
    Helper kecil untuk print rapi satu baris hasil metrik -- dipakai
    train.py & clean.py supaya format print konsisten di semua tempat,
    tidak perlu tulis ulang f-string yang sama berkali-kali.
    """
    return (
        f"MAE={metrics['mae']:.{precision}f} | "
        f"RMSE={metrics['rmse']:.{precision}f} | "
        f"Acc={metrics['accuracy']:.{precision}f} | "
        f"Off-by-1={metrics['off_by_one']:.{precision}f} | "
        f"QWK={metrics['qwk']:.{precision}f}"
    )