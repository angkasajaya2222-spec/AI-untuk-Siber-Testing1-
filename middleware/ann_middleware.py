"""
middleware/ann_middleware.py — Middleware Deteksi Brute-Force
──────────────────────────────────────────────────────────────
Jantung utama sistem. Middleware ini mencegat setiap request
HTTP ke endpoint auth, menjalankan pipeline lengkap:

  Request Masuk
       │
       ▼
  [1] Cek: Apakah IP sudah diblokir? ──── YA ──→ 429 (Block)
       │ TIDAK
       ▼
  [2] Cek: Apakah Redis tersedia?
       │ YA                  │ TIDAK
       ▼                     ▼
  [3a] Ekstrak 14 fitur   [3b] Fallback Rate Limiter
       │                       │
       ▼                       │
  [4] Inferensi ANN (<50ms)    │
       │                       │
       ▼                       │
  [5] Tentukan aksi:           │
    < 0.40  → PASS             │
    0.40-0.70 → CHALLENGE      │
    > 0.70  → BLOCK            │
       │ ◄─────────────────────┘
       ▼
  [6] Catat ke Security Log
       │
       ▼
  [7] Teruskan / Tolak Request
       │
       ▼
  [8] Tangkap response (sukses/gagal) → Update Redis
"""

import asyncio
import json
import logging
import time
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from config import settings
from core.feature_extractor import FeatureExtractor
from core.inference_engine import inference_engine
from core.redis_store import redis_store
from core.security_logger import ThreatAction, security_logger

logger = logging.getLogger(__name__)


class ANNBruteForceMiddleware(BaseHTTPMiddleware):
    """
    Middleware async berbasis ANN untuk deteksi brute-force real-time.

    Dipasang sebagai ASGI middleware di FastAPI/Starlette — 
    dieksekusi untuk SETIAP request masuk.
    
    Hanya aktif pada endpoint yang terdaftar di PROTECTED_PATHS.
    Endpoint lain dilewati langsung untuk efisiensi.
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.extractor = FeatureExtractor(redis_store)
        logger.info("[Middleware] ANNBruteForceMiddleware diinisialisasi.")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Entry point middleware — dipanggil untuk setiap request.
        """
        # ── Bypass: hanya proses endpoint yang dilindungi ─────────────────
        if not self._is_protected(request.url.path):
            return await call_next(request)

        ip       = self._get_ip(request)
        endpoint = request.url.path
        username = ""

        # ── Fase 1: Baca body untuk ekstrak username ───────────────────────
        # Body dibaca sekali, lalu "dikembalikan" agar handler bisa membacanya lagi
        body_bytes = await request.body()
        username   = await self._extract_username(request, body_bytes)

        # Rekonstruksi request agar bisa dibaca handler downstream
        async def receive():
            return {"type": "http.request", "body": body_bytes, "more_body": False}
        request = Request(request.scope, receive)

        # ── Fase 2: Cek blokir eksisting ─────────────────────────────────
        if await redis_store.is_blocked(ip):
            ttl = await redis_store.get_block_ttl(ip)
            event = security_logger.build_event(
                ip=ip, username=username, endpoint=endpoint,
                threat_score=1.0, action=ThreatAction.ALREADY_BLOCKED,
                user_agent=request.headers.get("user-agent", ""),
                block_ttl=ttl,
            )
            security_logger.log_event(event)
            return self._blocked_response(ttl)

        # ── Fase 3: Pilih jalur Redis vs Fallback ─────────────────────────
        if redis_store.is_available:
            return await self._redis_pipeline(
                request, call_next, ip, username, endpoint, body_bytes
            )
        else:
            return await self._fallback_pipeline(
                request, call_next, ip, username, endpoint
            )

    # ═══════════════════════════════════════════════════════════════════════
    # Pipeline Utama (Redis + ANN)
    # ═══════════════════════════════════════════════════════════════════════

    async def _redis_pipeline(
        self,
        request: Request,
        call_next: Callable,
        ip: str,
        username: str,
        endpoint: str,
        body_bytes: bytes,
    ) -> Response:
        """
        Pipeline lengkap: Redis → Feature Extraction → ANN → Mitigasi.
        """
        t_start = time.perf_counter()

        # 1. Catat request ke Redis (sebelum tahu hasilnya)
        await redis_store.record_request(
            ip=ip,
            username=username,
            success=True,  # Asumsi sukses dulu; diupdate setelah response
            user_agent=request.headers.get("user-agent", ""),
            accept_lang=request.headers.get("accept-language", ""),
            has_session_cookie=bool(request.cookies),
            endpoint=endpoint,
        )

        # 2. Ekstrak 14 fitur dari konteks request + riwayat Redis
        features = await self.extractor.extract(
            request=request,
            username=username,
            is_failed_attempt=False,
        )

        # 3. Inferensi ANN (target < 50ms)
        threat_score = await inference_engine.predict(features)

        total_ms = (time.perf_counter() - t_start) * 1000
        logger.debug(
            f"[Pipeline] IP={ip} score={threat_score:.3f} latency={total_ms:.1f}ms"
        )

        # 4. Tentukan dan eksekusi aksi mitigasi
        action = self._decide_action(threat_score)

        if action == ThreatAction.BLOCK:
            # Blokir IP dan langsung tolak
            await redis_store.block_ip(ip)
            ttl = settings.BLOCK_DURATION_S
            event = security_logger.build_event(
                ip=ip, username=username, endpoint=endpoint,
                threat_score=threat_score, action=ThreatAction.BLOCK,
                features=features,
                user_agent=request.headers.get("user-agent", ""),
                model_loaded=inference_engine._model_loaded,
                block_ttl=ttl,
            )
            security_logger.log_event(event)
            return self._blocked_response(ttl)

        elif action == ThreatAction.CHALLENGE:
            # Delay buatan + header challenge
            await asyncio.sleep(settings.CHALLENGE_DELAY_S)
            event = security_logger.build_event(
                ip=ip, username=username, endpoint=endpoint,
                threat_score=threat_score, action=ThreatAction.CHALLENGE,
                features=features,
                user_agent=request.headers.get("user-agent", ""),
                model_loaded=inference_engine._model_loaded,
            )
            security_logger.log_event(event)
            # Teruskan request tapi tambahkan header CAPTCHA
            response = await call_next(request)
            response.headers["X-Challenge-Required"] = "recaptcha"
            response.headers["X-Threat-Score"] = f"{threat_score:.3f}"
            await self._update_record_on_response(ip, username, response)
            return response

        else:  # PASS
            event = security_logger.build_event(
                ip=ip, username=username, endpoint=endpoint,
                threat_score=threat_score, action=ThreatAction.PASS,
                features=features,
                user_agent=request.headers.get("user-agent", ""),
                model_loaded=inference_engine._model_loaded,
            )
            security_logger.log_event(event)
            response = await call_next(request)
            await self._update_record_on_response(ip, username, response)
            return response

    # ═══════════════════════════════════════════════════════════════════════
    # Pipeline Fallback (Rate Limiter Standar)
    # ═══════════════════════════════════════════════════════════════════════

    async def _fallback_pipeline(
        self,
        request: Request,
        call_next: Callable,
        ip: str,
        username: str,
        endpoint: str,
    ) -> Response:
        """
        Fallback sederhana saat Redis/ANN tidak tersedia.
        Menggunakan in-memory counter untuk rate limiting.

        PRINSIP FAIL-OPEN: Jika semua sistem gagal, request tetap diizinkan
        (kecuali yang sangat jelas brute-force menurut counter sederhana).
        Lebih baik sedikit false-negative daripada memblokir user sah.
        """
        count = await redis_store.get_fallback_count(ip)

        if count > settings.FALLBACK_MAX_REQUESTS:
            logger.warning(
                f"[Fallback] IP {ip} melampaui batas "
                f"({count}/{settings.FALLBACK_MAX_REQUESTS}). Blokir sementara."
            )
            event = security_logger.build_event(
                ip=ip, username=username, endpoint=endpoint,
                threat_score=0.85, action=ThreatAction.BLOCK,
                user_agent=request.headers.get("user-agent", ""),
                model_loaded=False,
                note="fallback_rate_limiter",
            )
            security_logger.log_event(event)
            return self._blocked_response(settings.FALLBACK_WINDOW_S)

        event = security_logger.build_event(
            ip=ip, username=username, endpoint=endpoint,
            threat_score=0.0, action=ThreatAction.FALLBACK,
            user_agent=request.headers.get("user-agent", ""),
            model_loaded=False,
            note=f"fallback_counter={count}",
        )
        security_logger.log_event(event)
        return await call_next(request)

    # ═══════════════════════════════════════════════════════════════════════
    # Helper Methods
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _decide_action(score: float) -> ThreatAction:
        """
        Konversi skor probabilitas ke aksi konkret.

          < 0.40  → PASS      (aman)
          0.40–0.70 → CHALLENGE (mencurigakan, perlu verifikasi)
          > 0.70  → BLOCK     (sangat mencurigakan, blokir)
        """
        if score < settings.THRESHOLD_PASS:
            return ThreatAction.PASS
        elif score < settings.THRESHOLD_CHALLENGE:
            return ThreatAction.CHALLENGE
        else:
            return ThreatAction.BLOCK

    async def _update_record_on_response(
        self, ip: str, username: str, response: Response
    ) -> None:
        """
        Setelah response diterima, update catatan Redis
        apakah ini login sukses atau gagal.

        HTTP 200/201 → sukses login
        HTTP 401/403/422 → gagal login
        """
        success = response.status_code in (200, 201, 302)
        try:
            await redis_store.record_request(
                ip=ip,
                username=username,
                success=success,
            )
        except Exception:
            pass  # Jangan biarkan error logging merusak response

    @staticmethod
    async def _extract_username(request: Request, body_bytes: bytes) -> str:
        """
        Ekstrak username dari berbagai format request body.

        Mendukung:
        - application/json        : {"username": "...", "email": "..."}
        - application/x-www-form-urlencoded: username=...
        - Query parameter         : ?username=...
        """
        # Coba query params
        username = request.query_params.get("username", "")
        if username:
            return username[:100]

        content_type = request.headers.get("content-type", "")

        if "application/json" in content_type:
            try:
                body = json.loads(body_bytes)
                username = (
                    body.get("username")
                    or body.get("email")
                    or body.get("user")
                    or body.get("login")
                    or ""
                )
                return str(username)[:100]
            except (json.JSONDecodeError, AttributeError):
                pass

        elif "application/x-www-form-urlencoded" in content_type:
            try:
                from urllib.parse import parse_qs
                params = parse_qs(body_bytes.decode("utf-8", errors="ignore"))
                for key in ("username", "email", "user", "login"):
                    if key in params:
                        return params[key][0][:100]
            except Exception:
                pass

        return "unknown"

    @staticmethod
    def _is_protected(path: str) -> bool:
        """Cek apakah path ini perlu dilindungi middleware."""
        for protected in settings.PROTECTED_PATHS:
            if path.startswith(protected):
                return True
        return False

    @staticmethod
    def _get_ip(request: Request) -> str:
        """Dapatkan IP klien dengan memperhatikan proxy."""
        xff = request.headers.get("x-forwarded-for")
        if xff:
            return xff.split(",")[0].strip()
        xri = request.headers.get("x-real-ip")
        if xri:
            return xri.strip()
        if request.client:
            return request.client.host
        return "unknown"

    @staticmethod
    def _blocked_response(ttl_seconds: int = 900) -> JSONResponse:
        """Response HTTP 429 standar untuk IP yang diblokir."""
        return JSONResponse(
            status_code=429,
            content={
                "error": "too_many_requests",
                "message": (
                    "Akses Anda diblokir sementara karena aktivitas mencurigakan. "
                    f"Coba lagi dalam {ttl_seconds // 60} menit."
                ),
                "retry_after": ttl_seconds,
                "support": "Jika Anda yakin ini salah, hubungi support@example.com",
            },
            headers={
                "Retry-After": str(ttl_seconds),
                "X-Block-Reason": "brute_force_detected",
            },
        )
