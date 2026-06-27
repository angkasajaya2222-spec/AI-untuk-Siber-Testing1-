"""
models/ann_model.py — Arsitektur Jaringan Syaraf Tiruan
─────────────────────────────────────────────────────────
Feedforward Neural Network (FNN) ringan untuk inferensi
real-time. Berdasarkan paper:

  "Brute-Force Attack Detection on Computer Networks
   Using Artificial Neural Network"
  (Journal of AI and Engineering Applications, Feb 2026)

Arsitektur:
  Input(14) → Dense(64,ReLU) → Dropout(0.3)
            → Dense(32,ReLU) → Dropout(0.2)
            → Dense(1,Sigmoid)

Output: skor probabilitas ancaman [0.0 – 1.0]
"""

import torch
import torch.nn as nn
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BruteForceANN(nn.Module):
    """
    Model ANN untuk deteksi brute-force.

    Analogi: bayangkan ini sebagai "hakim" berlapis.
    Layer pertama menangkap pola kasar (banyak request cepat),
    layer kedua menyaring pola halus (variasi credential),
    output memberikan vonis: seberapa mencurigakan request ini.
    """

    def __init__(
        self,
        input_dim: int = 14,
        hidden1: int = 64,
        hidden2: int = 32,
        output_dim: int = 1,
        dropout1: float = 0.3,
        dropout2: float = 0.2,
    ):
        super().__init__()

        # ── Layer Definition ──────────────────────────────────
        self.network = nn.Sequential(
            # Layer 1: Ekstrak pola dasar dari fitur mentah
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),        # normalisasi agar training stabil
            nn.ReLU(),
            nn.Dropout(dropout1),

            # Layer 2: Kombinasikan pola menjadi representasi ancaman
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout2),

            # Output: satu angka → probabilitas "ini serangan"
            nn.Linear(hidden2, output_dim),
            nn.Sigmoid(),                   # squash ke [0.0, 1.0]
        )

        # Inisialisasi bobot Xavier untuk konvergensi lebih cepat
        self._init_weights()

    def _init_weights(self):
        """Inisialisasi bobot dengan Xavier Uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> float:
        """
        Inferensi tunggal — kembalikan float skor ancaman.
        Selalu jalankan dalam eval mode + no_grad untuk kecepatan.
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            return output.item()


# ── Model Factory ─────────────────────────────────────────────────────────────

def build_model(input_dim: int = 14) -> BruteForceANN:
    """Buat instance model baru."""
    return BruteForceANN(input_dim=input_dim)


def save_model(model: BruteForceANN, path: Path) -> None:
    """Simpan state_dict model ke disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": model.network[0].in_features,
        "architecture": str(model),
    }, path)
    logger.info(f"Model disimpan → {path}")


def load_model(path: Path, input_dim: int = 14) -> BruteForceANN:
    """
    Muat model dari disk dengan graceful error handling.
    Jika file tidak ada → kembalikan model baru (belum terlatih).
    """
    model = BruteForceANN(input_dim=input_dim)

    if not path.exists():
        logger.warning(
            f"[ANN] File model tidak ditemukan di {path}. "
            "Gunakan model baru (belum dilatih). Jalankan train.py dahulu!"
        )
        return model

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        logger.info(f"[ANN] Model berhasil dimuat dari {path}")
    except Exception as exc:
        logger.error(f"[ANN] Gagal memuat model: {exc}. Fallback ke model kosong.")

    return model
