"""
core/security_logger.py — Pencatatan Event Keamanan Terstruktur
────────────────────────────────────────────────────────────────
Semua event keamanan (blokir, tantangan, lolos) dicatat
dalam format JSON terstruktur untuk:
  - Analisis forensik
  - Pembuatan laporan
  - Integrasi SIEM (Splunk, ELK Stack, dll.)
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

from config import settings


class ThreatAction(str, Enum):
    PASS      = "PASS"       # skor < 0.40, request diizinkan
    CHALLENGE = "CHALLENGE"  # skor 0.40–0.70, minta verifikasi
    BLOCK     = "BLOCK"      # skor > 0.70, request diblokir
    FALLBACK  = "FALLBACK"   # AI down, gunakan rate-limiter standar
    ALREADY_BLOCKED = "ALREADY_BLOCKED"  # IP sudah dalam daftar blokir


@dataclass
class SecurityEvent:
    """Struktur event keamanan yang dicatat ke log."""
    # Identitas
    timestamp: float
    ip: str
    username: str
    endpoint: str

    # Keputusan AI
    threat_score: float
    action: str

    # Konteks request
    user_agent: str
    req_count_1min: int
    req_count_5min: int
    failure_rate: float
    unique_usernames: int
    interval_mean_ms: float

    # Metadata tambahan
    model_loaded: bool = True
    block_ttl: Optional[int] = None
    note: str = ""


class SecurityLogger:
    """
    Logger keamanan dengan dua output:
    1. File JSON line-delimited (untuk SIEM/parsing)
    2. Console logger terformat (untuk monitoring)
    """

    def __init__(self):
        self._setup()

    def _setup(self):
        log_path = Path(settings.LOG_PATH)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # File handler — format JSON per baris
        self._file_handler = logging.FileHandler(log_path, encoding="utf-8")
        self._file_handler.setFormatter(logging.Formatter("%(message)s"))

        # Logger khusus keamanan
        self._logger = logging.getLogger("security_events")
        self._logger.setLevel(logging.INFO)
        self._logger.addHandler(self._file_handler)
        self._logger.propagate = False

        # Console logger terpisah dengan warna
        self._console = logging.getLogger("security_console")

    def log_event(self, event: SecurityEvent) -> None:
        """Tulis event ke file log (JSON) dan console."""
        event_dict = asdict(event)
        event_dict["timestamp_iso"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(event.timestamp)
        )

        # Tulis ke file sebagai JSON per baris
        self._logger.info(json.dumps(event_dict, ensure_ascii=False))

        # Tampilkan di console dengan level yang sesuai
        msg = (
            f"[{event.action:17s}] IP={event.ip:20s} "
            f"user={event.username:20s} score={event.threat_score:.3f} "
            f"req1m={event.req_count_1min:4d} "
            f"fail%={event.failure_rate*100:5.1f}% "
            f"ep={event.endpoint}"
        )

        if event.action == ThreatAction.BLOCK:
            self._console.warning(f"🔴 {msg}")
        elif event.action == ThreatAction.CHALLENGE:
            self._console.info(f"🟡 {msg}")
        elif event.action == ThreatAction.PASS:
            self._console.debug(f"🟢 {msg}")
        elif event.action == ThreatAction.ALREADY_BLOCKED:
            self._console.warning(f"⛔ {msg}")
        else:
            self._console.info(f"⚪ {msg}")

    def build_event(
        self,
        ip: str,
        username: str,
        endpoint: str,
        threat_score: float,
        action: ThreatAction,
        features=None,
        user_agent: str = "",
        model_loaded: bool = True,
        block_ttl: Optional[int] = None,
        note: str = "",
    ) -> SecurityEvent:
        """Helper untuk membuat SecurityEvent dari komponen-komponen."""
        return SecurityEvent(
            timestamp=time.time(),
            ip=ip,
            username=username,
            endpoint=endpoint,
            threat_score=round(threat_score, 4),
            action=action.value,
            user_agent=user_agent[:200],  # batasi panjang UA
            req_count_1min=features.req_count_1min if features else 0,
            req_count_5min=features.req_count_5min if features else 0,
            failure_rate=round(features.failure_rate_1min, 4) if features else 0.0,
            unique_usernames=features.unique_usernames_1m if features else 0,
            interval_mean_ms=round(features.interval_mean_ms, 2) if features else 0.0,
            model_loaded=model_loaded,
            block_ttl=block_ttl,
            note=note,
        )


# Singleton global
security_logger = SecurityLogger()
