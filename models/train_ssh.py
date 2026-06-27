"""
models/train_ssh.py — Script Training Khusus Dataset SSH
──────────────────────────────────────────────────────────
Disesuaikan untuk dataset ssh_anomaly_dataset.csv dengan kolom:
  timestamp, source_ip, username, event_type, status, label, detail

Cara pakai:
  py -3.12 -m models.train_ssh --dataset data/ssh_anomaly_dataset.csv
"""

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from models.ann_model import BruteForceANN, save_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("train_ssh")


# ── Label Mapping ─────────────────────────────────────────────────────────────
# 0 = Normal, 1 = Serangan
LABEL_MAP = {
    "normal":                       0,
    "config_anomaly":               0,
    "brute_force":                  1,
    "brute_force_connection_issue": 1,
}


def engineer_features(df: pd.DataFrame):
    """
    Buat 14 fitur dari kolom SSH log.

    Dua jenis fitur:
    A) Per-event  : fitur dari satu baris log (event_type, status, jam)
    B) Per-IP     : statistik agregat perilaku satu IP di seluruh dataset
                    (berapa kali gagal login, berapa username berbeda, dll.)

    Analogi: seperti melihat "rekam jejak" setiap IP —
    bukan cuma satu kejadian, tapi keseluruhan polanya.
    """
    logger.info("Membangun 14 fitur dari dataset SSH...")
    df = df.copy()

    # ── A. Fitur Per-Event ────────────────────────────────────────────────────

    # [0] Apakah ini gagal login?
    df["is_failed_password"] = (df["event_type"] == "Failed password").astype(float)

    # [1] Apakah ini login berhasil?
    df["is_accepted_password"] = (df["event_type"] == "Accepted password").astype(float)

    # [2] Apakah koneksi terputus?
    df["is_disconnected"] = (df["event_type"] == "Disconnected").astype(float)

    # [3] Apakah ada eksekusi perintah? (anomali setelah masuk)
    df["is_command_executed"] = (df["event_type"] == "Command executed").astype(float)

    # [4] Apakah koneksi ditolak?
    df["is_connection_error"] = (df["event_type"] == "Connection error").astype(float)

    # [5] Jam kejadian (0.0 = tengah malam, 1.0 = 23:59)
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], infer_datetime_format=True)
        df["hour_norm"] = df["timestamp"].dt.hour / 24.0
    except Exception:
        logger.warning("Gagal parse timestamp, gunakan nilai default 0.5")
        df["hour_norm"] = 0.5

    # ── B. Fitur Per-IP (Statistik Agregat) ───────────────────────────────────
    logger.info("Menghitung statistik per IP...")

    ip_stats = df.groupby("source_ip").agg(
        ip_total_events  = ("event_type", "count"),
        ip_failed_count  = ("is_failed_password", "sum"),
        ip_success_count = ("is_accepted_password", "sum"),
        ip_refused_count = ("is_connection_error", "sum"),
        ip_unique_users  = ("username", "nunique"),
    ).reset_index()

    ip_stats["ip_failure_rate"] = ip_stats["ip_failed_count"]  / ip_stats["ip_total_events"]
    ip_stats["ip_success_rate"] = ip_stats["ip_success_count"] / ip_stats["ip_total_events"]
    ip_stats["ip_refused_rate"] = ip_stats["ip_refused_count"] / ip_stats["ip_total_events"]

    max_events = ip_stats["ip_total_events"].max()
    max_users  = ip_stats["ip_unique_users"].max()

    ip_stats["ip_event_count_norm"]  = ip_stats["ip_total_events"] / max(max_events, 1)
    ip_stats["ip_unique_users_norm"] = ip_stats["ip_unique_users"] / max(max_users, 1)

    # Flag volume tinggi (ciri khas brute-force)
    ip_stats["ip_is_high_volume"]    = (ip_stats["ip_total_events"] > 100).astype(float)
    ip_stats["ip_is_extreme_volume"] = (ip_stats["ip_total_events"] > 500).astype(float)

    df = df.merge(ip_stats[[
        "source_ip",
        "ip_event_count_norm",   # [6]
        "ip_failure_rate",       # [7]
        "ip_unique_users_norm",  # [8]
        "ip_success_rate",       # [9]
        "ip_refused_rate",       # [10]
        "ip_is_high_volume",     # [11]
        "ip_is_extreme_volume",  # [12]
    ]], on="source_ip", how="left")

    # [13] Seberapa sering username ini jadi target?
    user_freq = df["username"].value_counts(normalize=True)
    df["username_target_rate"] = df["username"].map(user_freq).fillna(0)

    # ── Susun 14 Fitur Final ──────────────────────────────────────────────────
    FEATURE_COLS = [
        "is_failed_password",    # 0
        "is_accepted_password",  # 1
        "is_disconnected",       # 2
        "is_command_executed",   # 3
        "is_connection_error",   # 4
        "hour_norm",             # 5
        "ip_event_count_norm",   # 6
        "ip_failure_rate",       # 7
        "ip_unique_users_norm",  # 8
        "ip_success_rate",       # 9
        "ip_refused_rate",       # 10
        "ip_is_high_volume",     # 11
        "ip_is_extreme_volume",  # 12
        "username_target_rate",  # 13
    ]

    X = df[FEATURE_COLS].fillna(0.0).clip(0.0, 1.0)
    logger.info(f"14 Fitur berhasil dibuat. Shape: {X.shape}")
    return X, FEATURE_COLS


def load_dataset(csv_path: str):
    """Muat CSV dan kembalikan X (fitur) dan y (label biner)."""
    logger.info(f"Memuat: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Shape: {df.shape}")

    df["label"] = df["label"].str.strip().str.lower()
    df["label_binary"] = df["label"].map(
        {k.lower(): v for k, v in LABEL_MAP.items()}
    )

    before = len(df)
    df = df.dropna(subset=["label_binary"])
    if len(df) < before:
        logger.warning(f"{before - len(df)} baris dibuang (label tidak dikenal)")

    y = df["label_binary"].astype(int)
    logger.info(f"Normal: {(y==0).sum():,} | Serangan: {(y==1).sum():,}")

    X, feature_names = engineer_features(df)
    return X.values.astype(np.float32), y.values.astype(np.float32), feature_names


def train(
    csv_path: str,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 0.001,
    model_out: Path = Path("saved_models/ann_model.pt"),
    scaler_out: Path = Path("saved_models/scaler.pkl"),
):
    """Pipeline lengkap: load → fitur → train → evaluasi → simpan."""

    X, y, feature_names = load_dataset(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    scaler_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "feature_names": feature_names}, scaler_out)
    logger.info(f"Scaler disimpan → {scaler_out}")

    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_te_t = torch.tensor(X_test,  dtype=torch.float32)
    y_te_t = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)

    train_dl = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=batch_size, shuffle=True,
    )

    model     = BruteForceANN(input_dim=14)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    logger.info(f"\n{'─'*55}")
    logger.info(f"Training: {epochs} epoch | batch={batch_size} | lr={lr}")
    logger.info(f"{'─'*55}")

    best_val_loss, best_state = float("inf"), None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for Xb, yb in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_te_t), y_te_t).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:03d}/{epochs} │ "
                f"Train Loss: {epoch_loss/len(train_dl):.4f} │ "
                f"Val Loss: {val_loss:.4f}"
            )

    # Evaluasi dengan model terbaik
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = model(X_te_t).numpy().flatten()
        preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)

    logger.info(f"\n{'═'*55}")
    logger.info(f"HASIL EVALUASI MODEL ANN")
    logger.info(f"{'═'*55}")
    logger.info(f"Akurasi: {acc*100:.2f}%")
    logger.info(f"\nClassification Report:\n"
                f"{classification_report(y_test, preds, target_names=['Normal','Serangan'])}")
    logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, preds)}")

    save_model(model, model_out)
    logger.info(f"\n✅ Model  → {model_out}")
    logger.info(f"✅ Scaler → {scaler_out}")
    logger.info(f"\nLangkah selanjutnya:")
    logger.info(f"  1. Jalankan Redis : docker run -d -p 6379:6379 redis:alpine")
    logger.info(f"  2. Jalankan server: uvicorn main:app --reload --port 8000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ANN untuk SSH Anomaly Dataset")
    parser.add_argument("--dataset",    required=True)
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=0.001)
    args = parser.parse_args()

    train(
        csv_path   = args.dataset,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
    )
