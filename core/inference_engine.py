"""
core/inference_engine.py — Mesin Inferensi ANN Real-time
──────────────────────────────────────────────────────────
Mengelola siklus hidup model ANN:
  - Load model dari disk saat startup
  - Bridge: mengkonversi fitur HTTP → fitur SSH (sesuai training)
  - Jalankan inferensi dengan timeout ketat (<50ms)
  - Fail-open: jika model gagal → gunakan heuristik sederhana
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch

from config import settings
from models.ann_model import BruteForceANN, load_model
from core.feature_extractor import ExtractedFeatures

logger = logging.getLogger(__name__)


def _http_to_ssh_features(f: ExtractedFeatures) -> np.ndarray:
    """
    Bridge: konversi fitur HTTP (dari middleware) ke
    fitur SSH (yang dipelajari model saat training).

    Mapping 14 fitur:
    SSH[0]  is_failed_password     ← failure_rate_1min > 0.3
    SSH[1]  is_accepted_password   ← 1 - failure_rate_1min
    SSH[2]  is_disconnected        ← 0 (N/A di HTTP)
    SSH[3]  is_command_executed    ← 0 (N/A di HTTP)
    SSH[4]  is_connection_error    ← 0 (N/A di HTTP)
    SSH[5]  hour_norm              ← time_of_day_norm
    SSH[6]  ip_event_count_norm    ← req_count_5min / 300
    SSH[7]  ip_failure_rate        ← failure_rate_1min
    SSH[8]  ip_unique_users_norm   ← unique_usernames_1m / 50
    SSH[9]  ip_success_rate        ← max(0, 1 - failure_rate_1min)
    SSH[10] ip_refused_rate        ← ua_is_generic * 0.3
    SSH[11] ip_is_high_volume      ← req_count_5min > 100
    SSH[12] ip_is_extreme_volume   ← req_count_5min > 500
    SSH[13] username_target_rate   ← unique_usernames_1m / max(unique_usernames_5m,1)
    """
    req_5min     = f.req_count_5min
    fail_rate    = f.failure_rate_1min
    uniq_1m      = f.unique_usernames_1m
    uniq_5m      = max(f.unique_usernames_5m, 1)

    return np.array([
        1.0 if fail_rate > 0.3 else fail_rate,          # [0]  is_failed_password
        max(0.0, 1.0 - fail_rate),                       # [1]  is_accepted_password
        0.0,                                             # [2]  is_disconnected
        0.0,                                             # [3]  is_command_executed
        float(f.ua_is_generic) * 0.5,                   # [4]  is_connection_error (proxy: UA aneh)
        f.time_of_day_norm,                              # [5]  hour_norm
        min(req_5min / 300.0, 1.0),                      # [6]  ip_event_count_norm
        fail_rate,                                       # [7]  ip_failure_rate
        min(uniq_1m / 50.0, 1.0),                        # [8]  ip_unique_users_norm
        max(0.0, 1.0 - fail_rate),                       # [9]  ip_success_rate
        float(f.ua_is_generic) * 0.3,                   # [10] ip_refused_rate (proxy)
        1.0 if req_5min > 100 else req_5min / 100.0,    # [11] ip_is_high_volume
        1.0 if req_5min > 500 else req_5min / 500.0,    # [12] ip_is_extreme_volume
        min(uniq_1m / uniq_5m, 1.0),                    # [13] username_target_rate
    ], dtype=np.float32)


class InferenceEngine:
    """
    Wrapper ringan di atas model ANN untuk inferensi produksi.

    Fitur utama:
    - Thread-safe: model disimpan dalam eval mode
    - Timeout: inferensi dipotong jika melebihi INFERENCE_TIMEOUT_MS
    - Fallback: skor heuristik jika ada error apapun
    - Scaler: StandardScaler diaplikasikan otomatis sebelum inferensi
    - Bridge: konversi fitur HTTP → fitur SSH (sesuai training)
    """

    def __init__(self):
        self._model: Optional[BruteForceANN] = None
        self._scaler = None
        self._feature_names: list[str] = []
        self._model_loaded: bool = False
        self._total_inferences: int = 0
        self._failed_inferences: int = 0
        self._latency_log: list[float] = []

    async def load(self) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)

    def _load_sync(self) -> None:
        try:
            scaler_path = settings.SCALER_PATH
            if scaler_path.exists():
                data = joblib.load(scaler_path)
                self._scaler = data.get("scaler")
                self._feature_names = data.get("feature_names", [])
                logger.info(f"[Inference] Scaler dimuat. Fitur training: {self._feature_names}")
            else:
                logger.warning(f"[Inference] Scaler tidak ditemukan di {scaler_path}.")

            self._model = load_model(
                path=settings.MODEL_PATH,
                input_dim=settings.MODEL_INPUT_DIM,
            )
            self._model.eval()
            self._model_loaded = True

            # Warm-up
            dummy = torch.zeros(1, settings.MODEL_INPUT_DIM)
            with torch.no_grad():
                _ = self._model(dummy)
            logger.info("[Inference] Model siap. Warm-up selesai.")

        except Exception as exc:
            logger.error(f"[Inference] GAGAL memuat model: {exc}. Fallback aktif.")
            self._model_loaded = False

    async def predict(self, features: ExtractedFeatures) -> float:
        """
        Inferensi utama — kembalikan skor ancaman [0.0 – 1.0].
        """
        if not self._model_loaded:
            return self._fallback_heuristic(features)

        loop = asyncio.get_event_loop()
        try:
            score = await asyncio.wait_for(
                loop.run_in_executor(None, self._infer_sync, features),
                timeout=settings.INFERENCE_TIMEOUT_MS / 1000.0,
            )
            return score

        except asyncio.TimeoutError:
            self._failed_inferences += 1
            logger.warning(f"[Inference] TIMEOUT. Fallback digunakan.")
            return self._fallback_heuristic(features)

        except Exception as exc:
            self._failed_inferences += 1
            logger.error(f"[Inference] Error: {exc}.")
            return self._fallback_heuristic(features)

    def _infer_sync(self, features: ExtractedFeatures) -> float:
        t_start = time.perf_counter()

        # Konversi fitur HTTP → fitur SSH (sesuai format training)
        feat_array = _http_to_ssh_features(features)

        # Terapkan scaler jika tersedia
        if self._scaler is not None:
            try:
                feat_array = self._scaler.transform(
                    feat_array.reshape(1, -1)
                ).flatten().astype(np.float32)
            except Exception as e:
                logger.warning(f"[Inference] Scaler gagal: {e}.")

        tensor = torch.tensor(feat_array, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            score = self._model(tensor).item()

        latency_ms = (time.perf_counter() - t_start) * 1000
        self._latency_log.append(latency_ms)
        if len(self._latency_log) > 100:
            self._latency_log.pop(0)
        self._total_inferences += 1

        logger.debug(f"[Inference] score={score:.3f} latency={latency_ms:.2f}ms")
        return float(score)

    @staticmethod
    def _fallback_heuristic(features: ExtractedFeatures) -> float:
        """Heuristik sederhana jika model tidak tersedia."""
        score = 0.35

        if features.req_count_1min > 20:  score += 0.20
        if features.req_count_1min > 50:  score += 0.20
        if features.failure_rate_1min > 0.8: score += 0.15
        if features.unique_usernames_1m > 5: score += 0.10
        if features.ua_is_generic:           score += 0.08
        if features.interval_std_ms < 50 and features.req_count_1min > 5:
            score += 0.10

        return min(score, 0.95)

    def get_stats(self) -> dict:
        return {
            "model_loaded": self._model_loaded,
            "total_inferences": self._total_inferences,
            "failed_inferences": self._failed_inferences,
            "avg_latency_ms": (
                round(sum(self._latency_log) / len(self._latency_log), 2)
                if self._latency_log else 0.0
            ),
            "p95_latency_ms": (
                round(sorted(self._latency_log)[int(len(self._latency_log) * 0.95)], 2)
                if len(self._latency_log) >= 10 else 0.0
            ),
        }


# Singleton global
inference_engine = InferenceEngine()
