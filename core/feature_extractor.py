"""
core/feature_extractor.py — Ekstraksi Fitur dari Request HTTP
───────────────────────────────────────────────────────────────
Mengubah request HTTP mentah menjadi vektor angka [0.0 – 1.0]
yang bisa dimengerti ANN.

Analogi: ini seperti "penerjemah" — mengubah bahasa HTTP
(header, IP, timestamp) ke bahasa matematika yang dipahami
jaringan syaraf.

Fitur yang diekstrak (14 dimensi):
  [0]  req_count_1min       — jumlah request dalam 1 menit
  [1]  req_count_5min       — jumlah request dalam 5 menit
  [2]  failure_rate_1min    — rasio gagal dalam 1 menit
  [3]  failure_rate_5min    — rasio gagal dalam 5 menit
  [4]  unique_usernames_1m  — variasi username per IP (1 menit)
  [5]  unique_usernames_5m  — variasi username per IP (5 menit)
  [6]  interval_mean_ms     — rata-rata jeda antar request
  [7]  interval_std_ms      — std dev jeda (rendah = bot)
  [8]  ua_entropy           — entropi string User-Agent
  [9]  ua_is_generic        — apakah UA generik/aneh?
  [10] has_session_cookie   — punya cookie sesi normal?
  [11] accept_lang_missing  — Accept-Language header absen?
  [12] time_of_day_norm     — jam (0.0=tengah malam, 1.0=23.99)
  [13] burst_score          — skor ledakan request tiba-tiba
"""

import math
import re
import time
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from fastapi import Request

from core.redis_store import RedisStore, RequestRecord

logger = logging.getLogger(__name__)


# ── Konfigurasi Fitur ─────────────────────────────────────────────────────────

FEATURE_DIM = 14

# User-Agent yang dianggap mencurigakan (bot/script biasa)
SUSPICIOUS_UA_PATTERNS = re.compile(
    r"(python-requests|curl|wget|go-http|java|okhttp|axios|"
    r"scrapy|bot|crawler|spider|nikto|sqlmap|hydra|medusa|"
    r"ncrack|patator)",
    re.IGNORECASE,
)

# Batas normalisasi (untuk scaling ke [0, 1])
MAX_REQ_1MIN = 100.0
MAX_REQ_5MIN = 300.0
MAX_UNIQUE_USERNAMES = 50.0
MAX_INTERVAL_MS = 60_000.0   # 60 detik
MAX_BURST_DIFF = 50.0


@dataclass
class ExtractedFeatures:
    """Representasi terstruktur dari fitur yang diekstrak."""
    # Raw values (untuk logging/debugging)
    req_count_1min: int = 0
    req_count_5min: int = 0
    failure_rate_1min: float = 0.0
    failure_rate_5min: float = 0.0
    unique_usernames_1m: int = 0
    unique_usernames_5m: int = 0
    interval_mean_ms: float = 0.0
    interval_std_ms: float = 0.0
    ua_entropy: float = 0.0
    ua_is_generic: bool = False
    has_session_cookie: bool = False
    accept_lang_missing: bool = False
    time_of_day_norm: float = 0.0
    burst_score: float = 0.0

    def to_numpy(self) -> np.ndarray:
        """Ubah ke array numpy yang dinormalisasi untuk input ANN."""
        return np.array([
            min(self.req_count_1min    / MAX_REQ_1MIN,    1.0),
            min(self.req_count_5min    / MAX_REQ_5MIN,    1.0),
            self.failure_rate_1min,                         # sudah [0,1]
            self.failure_rate_5min,                         # sudah [0,1]
            min(self.unique_usernames_1m / MAX_UNIQUE_USERNAMES, 1.0),
            min(self.unique_usernames_5m / MAX_UNIQUE_USERNAMES, 1.0),
            min(self.interval_mean_ms / MAX_INTERVAL_MS, 1.0),
            min(self.interval_std_ms  / MAX_INTERVAL_MS, 1.0),
            self.ua_entropy / 8.0,                          # entropy maks ~8
            float(self.ua_is_generic),
            float(not self.has_session_cookie),             # 1 = tidak punya cookie
            float(self.accept_lang_missing),
            self.time_of_day_norm,
            min(self.burst_score / MAX_BURST_DIFF, 1.0),
        ], dtype=np.float32)


# ── Helper Functions ──────────────────────────────────────────────────────────

def _shannon_entropy(s: str) -> float:
    """
    Hitung Shannon entropy dari string.

    Analogi: mengukur "keacakan" sebuah string.
    'aaaaaaa' = entropy rendah (bot sering kirim UA repetitif).
    'Mozilla/5.0 (Windows NT 10.0...' = entropy tinggi (user asli).
    """
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    return -sum((f/n) * math.log2(f/n) for f in freq.values())


def _compute_interval_stats(records: list[RequestRecord]) -> tuple[float, float]:
    """
    Hitung mean dan std dev jeda antar request (dalam ms).

    Bot biasanya sangat konsisten (std rendah, mean sangat rendah).
    Manusia memiliki interval yang lebih acak.
    """
    if len(records) < 2:
        return 0.0, 0.0

    timestamps = sorted(r.timestamp for r in records)
    intervals  = [(timestamps[i+1] - timestamps[i]) * 1000
                  for i in range(len(timestamps) - 1)]

    mean_ms = float(np.mean(intervals))
    std_ms  = float(np.std(intervals))
    return mean_ms, std_ms


def _is_generic_ua(user_agent: str) -> bool:
    """Cek apakah User-Agent mencurigakan atau terlalu generik."""
    if not user_agent or len(user_agent) < 10:
        return True
    return bool(SUSPICIOUS_UA_PATTERNS.search(user_agent))


def _compute_burst_score(records_1min: list[RequestRecord], records_5min: list[RequestRecord]) -> float:
    """
    Skor ledakan request: selisih antara 1 menit terakhir vs rata-rata 5 menit.
    Jika 1 menit terakhir jauh lebih tinggi → kemungkinan besar burst attack.
    """
    avg_per_min_5m = len(records_5min) / 5.0
    count_1m = len(records_1min)
    return max(0.0, count_1m - avg_per_min_5m)


# ── Main Feature Extractor ────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Mengekstrak vektor fitur dari request HTTP + riwayat Redis.

    Desain non-blocking: semua operasi Redis dilakukan secara async
    agar tidak memblokir event loop FastAPI.
    """

    def __init__(self, store: RedisStore):
        self.store = store

    async def extract(
        self,
        request: Request,
        username: str,
        is_failed_attempt: bool = False,
    ) -> ExtractedFeatures:
        """
        Ekstrasi penuh. Dipanggil SEBELUM request diteruskan ke handler.

        Args:
            request          : objek FastAPI Request
            username         : username dari body request
            is_failed_attempt: True jika ini percobaan gagal (dari response middleware)
        """
        ip          = self._get_client_ip(request)
        user_agent  = request.headers.get("user-agent", "")
        accept_lang = request.headers.get("accept-language", "")
        has_cookie  = bool(request.cookies)
        endpoint    = request.url.path

        # Ambil riwayat dari Redis (async, non-blocking)
        records_1min  = await self.store.get_records_in_window(ip, 60)
        records_5min  = await self.store.get_records_in_window(ip, 300)

        # Hitung fitur dari riwayat
        count_1m  = len(records_1min)
        count_5m  = len(records_5min)

        fail_rate_1m = self._failure_rate(records_1min)
        fail_rate_5m = self._failure_rate(records_5min)

        uniq_users_1m = len({r.username for r in records_1min})
        uniq_users_5m = len({r.username for r in records_5min})

        interval_mean, interval_std = _compute_interval_stats(records_5min)
        burst_score = _compute_burst_score(records_1min, records_5min)

        # Fitur header HTTP
        ua_entropy  = _shannon_entropy(user_agent)
        ua_generic  = _is_generic_ua(user_agent)
        lang_missing = len(accept_lang.strip()) == 0

        # Normalisasi waktu dalam hari
        hour_frac = (time.localtime().tm_hour * 3600
                     + time.localtime().tm_min * 60
                     + time.localtime().tm_sec) / 86400.0

        return ExtractedFeatures(
            req_count_1min=count_1m,
            req_count_5min=count_5m,
            failure_rate_1min=fail_rate_1m,
            failure_rate_5min=fail_rate_5m,
            unique_usernames_1m=uniq_users_1m,
            unique_usernames_5m=uniq_users_5m,
            interval_mean_ms=interval_mean,
            interval_std_ms=interval_std,
            ua_entropy=ua_entropy,
            ua_is_generic=ua_generic,
            has_session_cookie=has_cookie,
            accept_lang_missing=lang_missing,
            time_of_day_norm=hour_frac,
            burst_score=burst_score,
        )

    @staticmethod
    def _failure_rate(records: list[RequestRecord]) -> float:
        """Hitung rasio request gagal dalam sekumpulan record."""
        if not records:
            return 0.0
        failed = sum(1 for r in records if not r.success)
        return failed / len(records)

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        """
        Dapatkan IP asli klien, dengan memperhatikan proxy/load balancer.
        X-Forwarded-For dan X-Real-IP adalah header standar dari reverse proxy.
        """
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # Ambil IP paling kiri (IP klien asli, bukan proxy)
            return forwarded_for.split(",")[0].strip()
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()
        if request.client:
            return request.client.host
        return "unknown"
