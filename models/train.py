"""
models/train.py — Script Pelatihan Model ANN
──────────────────────────────────────────────
Melatih model BruteForceANN menggunakan dataset CSV.
Dataset yang didukung:
  - network-traffic-dataset (Kaggle) dengan kolom CICFlowMeter
  - Dataset khusus HTTP login dengan fitur yang diekstrak middleware

Cara pakai:
  python -m models.train --dataset data/network_traffic.csv
  python -m models.train --dataset data/custom_http_log.csv --epochs 50
"""

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from models.ann_model import BruteForceANN, save_model

# ── Setup Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("train")


# ── Konstanta Dataset ─────────────────────────────────────────────────────────

# Fitur hasil Feature Selection dari paper (7 fitur jaringan)
NETWORK_FEATURES = [
    "dst_port",
    "flow_duration",
    "tot_fwd_pkts",
    "tot_bwd_pkts",
    "flow_byts_s",
    "flow_pkts_s",
    "pkt_size_avg",
]

# Fitur HTTP tambahan untuk middleware web (7 fitur kontekstual)
HTTP_FEATURES = [
    "req_count_1min",        # jumlah request 1 menit terakhir
    "req_count_5min",        # jumlah request 5 menit terakhir
    "failure_rate",          # rasio login gagal
    "unique_usernames",      # variasi username per IP
    "interval_mean",         # rata-rata interval antar request (ms)
    "interval_std",          # standar deviasi interval (bot = std rendah)
    "header_anomaly_score",  # skor anomali header HTTP
]

# Label kelas dalam dataset jaringan → binary mapping
LABEL_COLUMN = "Label"
ATTACK_LABELS = {
    "NORMAL": 0,
    "Normal": 0,
    "BENIGN": 0,
    "Benign": 0,
    "Brute-force FTP": 1,
    "Brute-force SSH": 1,
    "Web Attack - Brute Force": 1,
    "Web Attack-Brute Force": 1,
    "FTP-BruteForce": 1,
    "SSH-BruteForce": 1,
}


# ── Data Loading & Preprocessing ──────────────────────────────────────────────

def load_and_preprocess(csv_path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Muat dataset, pilih fitur, dan encode label ke binary.

    Returns:
        X         : array fitur [n_samples, n_features]
        y         : array label binary [n_samples]
        feat_names: nama fitur yang digunakan
    """
    logger.info(f"Memuat dataset dari: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Kolom tersedia: {list(df.columns)}")

    # ── Pilih fitur yang tersedia di dataset ──────────────────────────────
    all_possible = NETWORK_FEATURES + HTTP_FEATURES
    available = [f for f in all_possible if f in df.columns]

    if len(available) < 3:
        raise ValueError(
            f"Dataset tidak memiliki cukup fitur yang dikenal.\n"
            f"Fitur dibutuhkan: {all_possible}\n"
            f"Fitur tersedia: {list(df.columns)}"
        )

    logger.info(f"Fitur digunakan ({len(available)}): {available}")

    X = df[available].copy()

    # ── Label Encoding ────────────────────────────────────────────────────
    if LABEL_COLUMN in df.columns:
        df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip()
        y = df[LABEL_COLUMN].map(ATTACK_LABELS)

        # Coba pemetaan otomatis jika label tidak dikenali
        unmapped = y.isna().sum()
        if unmapped > 0:
            logger.warning(
                f"{unmapped} baris tidak dapat dipetakan. "
                "Label unik: " + str(df[LABEL_COLUMN].unique())
            )
            # Fallback: label yang bukan 0 → anggap serangan
            y = y.fillna(1).astype(int)
        else:
            y = y.astype(int)
    else:
        # Jika tidak ada kolom label, buat label dummy (untuk testing)
        logger.warning("Kolom 'Label' tidak ditemukan. Membuat label dummy (0=normal).")
        y = pd.Series(np.zeros(len(df), dtype=int))

    # ── Bersihkan nilai tak hingga & NaN ──────────────────────────────────
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    X = X.clip(lower=-1e9, upper=1e9)   # clamp outlier ekstrem

    logger.info(f"Distribusi label — Normal: {(y==0).sum()}, Serangan: {(y==1).sum()}")

    return X.values.astype(np.float32), y.values.astype(np.float32), available


# ── Training Loop ─────────────────────────────────────────────────────────────

def train(
    csv_path: str,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 0.001,
    model_out: Path = Path("saved_models/ann_model.pt"),
    scaler_out: Path = Path("saved_models/scaler.pkl"),
) -> None:
    """
    Pipeline lengkap: load → preprocess → train → evaluate → simpan.
    """
    # 1. Load data
    X, y, feature_names = load_and_preprocess(csv_path)

    # 2. Train/test split (80:20 — sesuai metodologi paper)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. Normalisasi fitur (StandardScaler)
    #    PENTING: scaler dilatih HANYA pada data training
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Simpan scaler untuk dipakai saat inferensi
    scaler_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "feature_names": feature_names}, scaler_out)
    logger.info(f"Scaler disimpan → {scaler_out}")

    # 4. Buat DataLoader PyTorch
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_test_t  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # 5. Inisialisasi model
    input_dim = X_train.shape[1]
    model = BruteForceANN(input_dim=input_dim)
    logger.info(f"Arsitektur model:\n{model}")
    logger.info(f"Total parameter: {sum(p.numel() for p in model.parameters()):,}")

    # 6. Loss & Optimizer
    #    BCELoss = Binary Cross Entropy — cocok untuk klasifikasi biner
    #    Analogi: mengukur seberapa "salah" prediksi model di setiap epoch
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=True
    )

    # 7. Training loop
    logger.info(f"\n{'─'*50}")
    logger.info(f"Mulai training: {epochs} epoch, batch={batch_size}, lr={lr}")
    logger.info(f"{'─'*50}")

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        # ── Training phase ────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_dl:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_dl)

        # ── Validation phase ──────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_preds = model(X_test_t)
            val_loss  = criterion(val_preds, y_test_t).item()

        scheduler.step(val_loss)

        # Simpan model terbaik (early stopping implicit)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:03d}/{epochs} │ "
                f"Train Loss: {avg_train_loss:.4f} │ "
                f"Val Loss: {val_loss:.4f}"
            )

    # 8. Evaluasi final
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        test_probs = model(X_test_t).numpy().flatten()
        test_preds = (test_probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, test_preds)
    logger.info(f"\n{'═'*50}")
    logger.info(f"HASIL EVALUASI MODEL")
    logger.info(f"{'═'*50}")
    logger.info(f"Akurasi: {acc*100:.2f}%")
    logger.info(f"\nClassification Report:\n{classification_report(y_test, test_preds, target_names=['Normal','Serangan'])}")
    logger.info(f"\nConfusion Matrix:\n{confusion_matrix(y_test, test_preds)}")

    # 9. Simpan model
    save_model(model, model_out)
    logger.info(f"\n✅ Training selesai. Model → {model_out}")


# ── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latih ANN Brute-Force Detector")
    parser.add_argument("--dataset", required=True, help="Path ke file CSV dataset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    train(
        csv_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
