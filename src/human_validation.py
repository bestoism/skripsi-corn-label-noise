import numpy as np
import pandas as pd

from src import config

VALID_VERDICTS = {"noise", "not_noise", "ambiguous"}


def _load_validated_sample():
    try:
        # sep=None + engine="python" -> auto-detect delimiter (koma ATAU titik
        # koma). Perlu karena Excel dengan locale Indonesia otomatis mengganti
        # delimiter CSV jadi ';' saat file di-save ulang -- kita tidak bisa
        # kontrol software apa yang dipakai user untuk isi validasi manual,
        # jadi baca-nya yang harus fleksibel.
        df = pd.read_csv(config.HUMAN_VALIDATION_FILE, sep=None, engine="python")
    except FileNotFoundError:
        print(f"⚠️ File belum ada: {config.HUMAN_VALIDATION_FILE}")
        print("   Jalankan clean.py dulu untuk men-generate file sample-nya.")
        return None

    df["human_verdict"] = df["human_verdict"].astype(str).str.strip().str.lower()
    return df


def _check_completeness(df):
    """
    Cek dua hal: (1) tidak ada baris kosong, (2) tidak ada nilai typo di luar
    3 kategori valid. Versi lama cuma cek kosong -- typo seperti 'noize' atau
    'ambigous' akan lolos diam-diam dan bikin hitungan agreement rate salah
    tanpa error apa pun.
    """
    empty_mask = df["human_verdict"].isin(["", "nan", "none"]) | df["human_verdict"].isna()
    n_empty = empty_mask.sum()

    if n_empty > 0:
        print(f"⚠️ Masih ada {n_empty} baris yang belum diisi 'human_verdict'.")
        print("   Isi manual semuanya, save, lalu jalankan lagi.")
        return False

    invalid_mask = ~df["human_verdict"].isin(VALID_VERDICTS)
    if invalid_mask.any():
        bad_values = df.loc[invalid_mask, "human_verdict"].unique().tolist()
        bad_rows = df.index[invalid_mask].tolist()
        print(f"⚠️ Ada nilai 'human_verdict' yang tidak dikenali: {bad_values}")
        print(f"   Baris ke-{bad_rows} (index dari 0). Nilai yang valid hanya:")
        print(f"   {sorted(VALID_VERDICTS)} -- perbaiki lalu jalankan lagi.")
        return False

    return True


def compute_agreement():
    """
    Jalankan SETELAH mengisi kolom 'human_verdict' secara manual di
    HUMAN_VALIDATION_FILE (lewat Excel/Google Sheets), nilai valid:
      - 'noise'     : label aslinya memang salah
      - 'not_noise' : label aslinya sudah benar (cleanlab salah tebak)
      - 'ambiguous' : ulasan sarkas/ambigu, sulit dinilai objektif

    Melaporkan agreement rate SECARA KESELURUHAN dan PER-BIN rating_diff --
    breakdown per-bin ini penting karena severity-aware pruning ditentukan
    berdasarkan rating_diff, jadi kita perlu tahu apakah agreement manusia
    juga berbeda antar bin (mis. apakah rating_diff=1 memang lebih sering
    'not_noise'/ambiguous dibanding rating_diff>=3, yang akan jadi bukti
    langsung yang mendukung/menolak asumsi severity-aware pruning di Bab 4).
    """
    df = _load_validated_sample()
    if df is None:
        return None

    if not _check_completeness(df):
        return None

    counts = df["human_verdict"].value_counts()
    total = len(df)
    agree = counts.get("noise", 0)
    ambiguous = counts.get("ambiguous", 0)
    disagree = counts.get("not_noise", 0)

    print("=" * 60)
    print(" HASIL VALIDASI MANUSIA vs CLEANLAB ")
    print("=" * 60)
    print(f"Total sample direview : {total}")
    print(f"Setuju (memang noise) : {agree} ({agree/total*100:.1f}%)")
    print(f"Ambigu                : {ambiguous} ({ambiguous/total*100:.1f}%)")
    print(f"Tidak setuju          : {disagree} ({disagree/total*100:.1f}%)")

    # ---- breakdown per-bin rating_diff (bukti langsung utk asumsi severity-aware) ----
    print("\n📊 Agreement rate per rating_diff (mendukung/menolak asumsi severity-aware pruning):")
    breakdown_rows = []
    for diff_val, group in df.groupby("rating_diff"):
        n = len(group)
        vc = group["human_verdict"].value_counts()
        row = {
            "rating_diff": diff_val,
            "n_sample": n,
            "pct_noise": round(vc.get("noise", 0) / n * 100, 1),
            "pct_not_noise": round(vc.get("not_noise", 0) / n * 100, 1),
            "pct_ambiguous": round(vc.get("ambiguous", 0) / n * 100, 1),
        }
        breakdown_rows.append(row)
        print(f"   diff={diff_val}: n={n} | noise={row['pct_noise']}% | "
              f"not_noise={row['pct_not_noise']}% | ambiguous={row['pct_ambiguous']}%")

    breakdown_df = pd.DataFrame(breakdown_rows).sort_values("rating_diff")

    print("-" * 60)
    print("Acuan pembanding (Northcutt et al., 2021, ImageNet): ~58% sample")
    print("yang direview terbukti benar-benar issue -- ini acuan wajar, bukan")
    print("standar mutlak yang harus dicapai (skala dan domain berbeda).")
    print("=" * 60)

    df.to_csv(config.HUMAN_VALIDATION_RESULT_FILE, index=False)
    print(f"\n💾 Hasil lengkap -> {config.HUMAN_VALIDATION_RESULT_FILE}")

    result = {
        "agree": int(agree),
        "ambiguous": int(ambiguous),
        "disagree": int(disagree),
        "total": int(total),
        "agreement_rate": round(agree / total, 4),
        "breakdown_by_rating_diff": breakdown_df.to_dict(orient="records"),
    }
    return result


if __name__ == "__main__":
    compute_agreement()