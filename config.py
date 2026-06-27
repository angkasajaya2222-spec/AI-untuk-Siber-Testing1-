"""
config.py — Pusat Konfigurasi Seluruh Sistem
─────────────────────────────────────────────
Semua konstanta, ambang batas, dan parameter
sistem dikontrol dari sini agar mudah di-tuning
tanpa menyentuh logika bisnis.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    # ── Aplikasi ──────────────────────────────
    APP_NAME: str = "ANN Brute-Force Detection Middleware"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # ── Redis ─────────────────────────────────
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""
    REDIS_TIMEOUT: float = 0.5          # detik — jika Redis timeout → fail-open

    # ── Sliding Window (Feature Extraction) ───
    WINDOW_1MIN: int = 60               # detik
    WINDOW_5MIN: int = 300              # detik
    WINDOW_15MIN: int = 900             # detik
    MAX_HISTORY_SIZE: int = 500         # maks record per IP di Redis

    # ── Model ANN ─────────────────────────────
    MODEL_PATH: Path = Path("saved_models/ann_model.pt")
    SCALER_PATH: Path = Path("saved_models/scaler.pkl")
    MODEL_INPUT_DIM: int = 14           # dimensi fitur vektor
    MODEL_HIDDEN_1: int = 64
    MODEL_HIDDEN_2: int = 32
    MODEL_OUTPUT_DIM: int = 1           # sigmoid → skor 0.0–1.0
    INFERENCE_TIMEOUT_MS: float = 45.0  # target <50ms

    # ── Threshold Mitigasi ────────────────────
    THRESHOLD_PASS: float = 0.40        # < 0.40 → Izinkan
    THRESHOLD_CHALLENGE: float = 0.70   # 0.40–0.70 → CAPTCHA / delay
    # > 0.70 → Blokir

    # ── Blokir Temporer ───────────────────────
    BLOCK_DURATION_S: int = 900         # 15 menit
    CHALLENGE_DELAY_S: float = 2.0      # delay buatan saat skor menengah

    # ── Rate Limit Fallback (tanpa Redis/AI) ──
    FALLBACK_MAX_REQUESTS: int = 10     # maks request per window
    FALLBACK_WINDOW_S: int = 60

    # ── Endpoint yang Dilindungi ──────────────
    PROTECTED_PATHS: list[str] = [
        "/login", "/auth", "/token",
        "/api/auth", "/api/login",
        "/signin", "/api/signin",
    ]

    # ── Logging ───────────────────────────────
    LOG_PATH: str = "logs/security.log"
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton global
settings = Settings()
