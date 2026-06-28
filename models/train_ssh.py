"""
models/train_ssh.py — Pipeline Training Lengkap dengan SMOTE + Bayesian Optimization
═══════════════════════════════════════════════════════════════════════════════════════
Pipeline ini mencakup:
  1. Load & Feature Engineering (14 fitur dari log SSH)
  2. SMOTE — menyeimbangkan kelas Normal vs Serangan
  3. Bayesian Optimization — mencari hyperparameter ANN terbaik
  4. Training ANN dengan hyperparameter optimal
  5. Perbandingan: ANN vs Random Forest vs Logistic Regression
  6. Simpan model terbaik

Cara pakai:
  py -3.12 -m models.train_ssh --dataset data/ssh_anomaly_dataset.csv
"""

import argparse
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import optuna
from optuna.samplers import TPESampler

from models.ann_model import BruteForceANN, save_model

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("train_ssh")

LABEL_MAP = {
    "normal":                       0,
    "config_anomaly":               0,
    "brute_force":                  1,
    "brute_force_connection_issue": 1,
}


def engineer_features(df):
    logger.info("Membangun 14 fitur dari dataset SSH...")
    df = df.copy()

    df["is_failed_password"]   = (df["event_type"] == "Failed password").astype(float)
    df["is_accepted_password"] = (df["event_type"] == "Accepted password").astype(float)
    df["is_disconnected"]      = (df["event_type"] == "Disconnected").astype(float)
    df["is_command_executed"]  = (df["event_type"] == "Command executed").astype(float)
    df["is_connection_error"]  = (df["event_type"] == "Connection error").astype(float)

    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour_norm"] = df["timestamp"].dt.hour / 24.0
    except Exception:
        df["hour_norm"] = 0.5

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
    ip_stats["ip_is_high_volume"]    = (ip_stats["ip_total_events"] > 100).astype(float)
    ip_stats["ip_is_extreme_volume"] = (ip_stats["ip_total_events"] > 500).astype(float)

    df = df.merge(ip_stats[[
        "source_ip", "ip_event_count_norm", "ip_failure_rate",
        "ip_unique_users_norm", "ip_success_rate", "ip_refused_rate",
        "ip_is_high_volume", "ip_is_extreme_volume",
    ]], on="source_ip", how="left")

    user_freq = df["username"].value_counts(normalize=True)
    df["username_target_rate"] = df["username"].map(user_freq).fillna(0)

    FEATURE_COLS = [
        "is_failed_password", "is_accepted_password", "is_disconnected",
        "is_command_executed", "is_connection_error", "hour_norm",
        "ip_event_count_norm", "ip_failure_rate", "ip_unique_users_norm",
        "ip_success_rate", "ip_refused_rate", "ip_is_high_volume",
        "ip_is_extreme_volume", "username_target_rate",
    ]

    X = df[FEATURE_COLS].fillna(0.0).clip(0.0, 1.0)
    logger.info(f"14 Fitur berhasil dibuat. Shape: {X.shape}")
    return X, FEATURE_COLS


def load_dataset(csv_path):
    logger.info(f"Memuat: {csv_path}")
    df = pd.read_csv(csv_path)
    df["label"] = df["label"].str.strip().str.lower()
    df["label_binary"] = df["label"].map({k.lower(): v for k, v in LABEL_MAP.items()})
    df = df.dropna(subset=["label_binary"])
    y = df["label_binary"].astype(int)
    logger.info(f"Normal: {(y==0).sum():,} | Serangan: {(y==1).sum():,}")
    X, feature_names = engineer_features(df)
    return X.values.astype(np.float32), y.values.astype(np.float32), feature_names


def apply_smote(X_train, y_train):
    """
    SMOTE — Synthetic Minority Oversampling Technique.
    Membuat data sintetis untuk kelas Normal agar seimbang dengan Serangan.
    Sebelum: Normal=8.86%, Serangan=90.25%
    Sesudah: Normal=50%,   Serangan=50%
    """
    logger.info("\n── TAHAP 2: SMOTE ──────────────────────────────────────")
    logger.info(f"Sebelum — Normal: {(y_train==0).sum():,} | Serangan: {(y_train==1).sum():,}")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    logger.info(f"Sesudah — Normal: {(y_res==0).sum():,} | Serangan: {(y_res==1).sum():,}")
    logger.info(f"Data training seimbang. Total: {len(X_res):,} sampel")
    return X_res.astype(np.float32), y_res.astype(np.float32)


def bayesian_optimization(X_train, y_train, X_val, y_val, n_trials=15):
    """
    Bayesian Optimization dengan Optuna (TPE Sampler).
    Mencari hyperparameter ANN terbaik secara cerdas.

    Parameter yang dioptimasi:
    - hidden1, hidden2 : ukuran hidden layer
    - lr               : learning rate
    - dropout1/2       : tingkat dropout
    - batch_size       : ukuran batch training
    """
    logger.info("\n── TAHAP 3: BAYESIAN OPTIMIZATION ──────────────────────")
    logger.info(f"Menjalankan {n_trials} trial pencarian hyperparameter...")

    X_tr_t  = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t  = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val,   dtype=torch.float32)
    y_val_t = torch.tensor(y_val,   dtype=torch.float32).unsqueeze(1)

    def objective(trial):
        h1  = trial.suggest_categorical("hidden1",    [32, 64, 128, 256])
        h2  = trial.suggest_categorical("hidden2",    [16, 32, 64, 128])
        lr  = trial.suggest_float("lr",               1e-4, 1e-2, log=True)
        d1  = trial.suggest_float("dropout1",         0.1, 0.5)
        d2  = trial.suggest_float("dropout2",         0.1, 0.4)
        bs  = trial.suggest_categorical("batch_size", [32, 64, 128])

        model = BruteForceANN(input_dim=14, hidden1=h1, hidden2=h2, dropout1=d1, dropout2=d2)
        opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        dl    = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=bs, shuffle=True)

        model.train()
        for _ in range(20):
            for Xb, yb in dl:
                opt.zero_grad()
                nn.BCELoss()(model(Xb), yb).backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = nn.BCELoss()(model(X_val_t), y_val_t).item()
        return val_loss

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    logger.info(f"Hyperparameter terbaik ditemukan:")
    for k, v in best.items():
        logger.info(f"   {k:15s} = {v}")
    return best


def train_ann(X_train, y_train, X_test, y_test, best_params, epochs=50):
    logger.info("\n── TAHAP 4: TRAINING ANN FINAL ─────────────────────────")

    h1 = best_params.get("hidden1",    64)
    h2 = best_params.get("hidden2",    32)
    lr = best_params.get("lr",         0.001)
    d1 = best_params.get("dropout1",   0.3)
    d2 = best_params.get("dropout2",   0.2)
    bs = best_params.get("batch_size", 64)

    logger.info(f"Arsitektur: Input(14)→Dense({h1},ReLU)→Dense({h2},ReLU)→Sigmoid(1)")

    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_te_t = torch.tensor(X_test,  dtype=torch.float32)
    y_te_t = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)

    dl    = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=bs, shuffle=True)
    model = BruteForceANN(input_dim=14, hidden1=h1, hidden2=h2, dropout1=d1, dropout2=d2)
    crit  = nn.BCELoss()
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    best_val, best_state = float("inf"), None
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for Xb, yb in dl:
            opt.zero_grad()
            loss = crit(model(Xb), yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_loss = crit(model(X_te_t), y_te_t).item()

        avg = ep_loss / len(dl)
        history["train_loss"].append(avg)
        history["val_loss"].append(val_loss)
        sched.step(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:03d}/{epochs} │ Train: {avg:.4f} │ Val: {val_loss:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = (model(X_te_t).numpy().flatten() >= 0.5).astype(int)

    return model, preds, history


def compare_models(X_train, y_train, X_test, y_test, ann_preds):
    """Bandingkan ANN vs Random Forest vs Logistic Regression."""
    logger.info("\n── TAHAP 5: PERBANDINGAN MODEL ──────────────────────────")
    results = {}

    logger.info("Melatih Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_p = rf.predict(X_test)
    cr_rf = classification_report(y_test, rf_p, output_dict=True)
    results["Random Forest"] = {
        "accuracy":  accuracy_score(y_test, rf_p),
        "precision": cr_rf["weighted avg"]["precision"],
        "recall":    cr_rf["weighted avg"]["recall"],
        "f1_score":  cr_rf["weighted avg"]["f1-score"],
    }

    logger.info("Melatih Logistic Regression...")
    lr_m = LogisticRegression(max_iter=1000, random_state=42)
    lr_m.fit(X_train, y_train)
    lr_p = lr_m.predict(X_test)
    cr_lr = classification_report(y_test, lr_p, output_dict=True)
    results["Logistic Regression"] = {
        "accuracy":  accuracy_score(y_test, lr_p),
        "precision": cr_lr["weighted avg"]["precision"],
        "recall":    cr_lr["weighted avg"]["recall"],
        "f1_score":  cr_lr["weighted avg"]["f1-score"],
    }

    cr_ann = classification_report(y_test, ann_preds, output_dict=True)
    results["ANN (Model Kami)"] = {
        "accuracy":  accuracy_score(y_test, ann_preds),
        "precision": cr_ann["weighted avg"]["precision"],
        "recall":    cr_ann["weighted avg"]["recall"],
        "f1_score":  cr_ann["weighted avg"]["f1-score"],
    }

    logger.info(f"\n{'═'*68}")
    logger.info(f"  TABEL PERBANDINGAN MODEL")
    logger.info(f"{'═'*68}")
    logger.info(f"  {'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    logger.info(f"  {'─'*63}")
    for name, m in results.items():
        logger.info(
            f"  {name:<25} "
            f"{m['accuracy']*100:>9.2f}% "
            f"{m['precision']*100:>9.2f}% "
            f"{m['recall']*100:>7.2f}% "
            f"{m['f1_score']*100:>7.2f}%"
        )
    logger.info(f"{'═'*68}")
    return results


def train(
    csv_path: str,
    epochs: int = 50,
    n_trials: int = 15,
    model_out: Path = Path("saved_models/ann_model.pt"),
    scaler_out: Path = Path("saved_models/scaler.pkl"),
):
    logger.info("=" * 68)
    logger.info("  ANN BRUTE-FORCE DETECTION — PIPELINE TRAINING LENGKAP")
    logger.info("=" * 68)

    logger.info("\n── TAHAP 1: LOAD DATA & FEATURE ENGINEERING ────────────")
    X, y, feature_names = load_dataset(csv_path)

    # Split SEBELUM SMOTE — test set harus data asli
    X_train_raw, X_test, y_train_raw, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_raw = scaler.fit_transform(X_train_raw)
    X_test      = scaler.transform(X_test)

    scaler_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "feature_names": feature_names}, scaler_out)
    logger.info(f"Scaler disimpan → {scaler_out}")

    # SMOTE pada training set saja
    X_train_smote, y_train_smote = apply_smote(X_train_raw, y_train_raw)

    # Split validation untuk Bayesian Opt
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_smote, y_train_smote,
        test_size=0.15, random_state=42, stratify=y_train_smote
    )

    best_params = bayesian_optimization(X_tr, y_tr, X_val, y_val, n_trials=n_trials)

    model, ann_preds, history = train_ann(
        X_train_smote, y_train_smote, X_test, y_test, best_params, epochs=epochs
    )

    acc = accuracy_score(y_test, ann_preds)
    logger.info(f"\n{'='*68}")
    logger.info(f"  HASIL EVALUASI MODEL ANN")
    logger.info(f"{'='*68}")
    logger.info(f"  Akurasi: {acc*100:.4f}%")
    logger.info(f"\n{classification_report(y_test, ann_preds, target_names=['Normal','Serangan'])}")
    logger.info(f"  Confusion Matrix:\n{confusion_matrix(y_test, ann_preds)}")

    compare_models(X_train_smote, y_train_smote, X_test, y_test, ann_preds)

    save_model(model, model_out)
    logger.info(f"\n  ✅ Training selesai!")
    logger.info(f"  ✅ Model  → {model_out}")
    logger.info(f"  ✅ Scaler → {scaler_out}")
    logger.info(f"\n  Jalankan server: py -3.12 -m uvicorn main:app --reload --port 8000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  required=True)
    parser.add_argument("--epochs",   type=int, default=50)
    parser.add_argument("--n-trials", type=int, default=15)
    args = parser.parse_args()
    train(csv_path=args.dataset, epochs=args.epochs, n_trials=args.n_trials)
