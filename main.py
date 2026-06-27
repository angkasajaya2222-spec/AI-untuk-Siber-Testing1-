"""
main.py — Aplikasi FastAPI dengan ANN Middleware
─────────────────────────────────────────────────
Entry point aplikasi. Menggabungkan semua komponen:
  - FastAPI app
  - ANNBruteForceMiddleware
  - Endpoint demo (login, register, admin)
  - Health check & monitoring endpoint

Cara jalankan:
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import settings
from core.inference_engine import inference_engine
from core.redis_store import redis_store
from core.security_logger import security_logger
from middleware.ann_middleware import ANNBruteForceMiddleware

# ── Logging Setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ── Lifecycle: Startup & Shutdown ──────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Mengelola sumber daya aplikasi:
    - Startup: muat model ANN, koneksi Redis
    - Shutdown: tutup koneksi dengan bersih
    """
    logger.info("═" * 60)
    logger.info(f"  {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info("═" * 60)

    # Muat model ANN (async — tidak memblokir startup)
    logger.info("[Startup] Memuat model ANN...")
    await inference_engine.load()

    # Test koneksi Redis
    logger.info("[Startup] Menghubungkan ke Redis...")
    await redis_store._get_client()

    logger.info("[Startup] Semua komponen siap. Server berjalan.")
    logger.info(f"[Startup] Endpoint dilindungi: {settings.PROTECTED_PATHS}")
    logger.info("─" * 60)

    yield  # Aplikasi berjalan di sini

    # Shutdown
    logger.info("[Shutdown] Menutup koneksi...")
    await redis_store.close()
    logger.info("[Shutdown] Selesai.")


# ── Inisialisasi FastAPI ────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Middleware Web Application dengan ANN untuk deteksi real-time "
        "percobaan brute-force dan credential stuffing."
    ),
    lifespan=lifespan,
)

# ── Pasang Middleware ───────────────────────────────────────────────────────
# URUTAN PENTING: Middleware dieksekusi dari bawah ke atas
app.add_middleware(ANNBruteForceMiddleware)


# ═══════════════════════════════════════════════════════════════════════════
# Request/Response Models
# ═══════════════════════════════════════════════════════════════════════════

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    message: str

class RegisterRequest(BaseModel):
    username: str
    password: str
    email: str

class AdminUnblockRequest(BaseModel):
    ip: str
    admin_key: str  # Dalam produksi: gunakan JWT/OAuth yang proper


# ═══════════════════════════════════════════════════════════════════════════
# Endpoint Otentikasi (Dilindungi Middleware)
# ═══════════════════════════════════════════════════════════════════════════

# Database pengguna demo (ganti dengan database nyata di produksi!)
DEMO_USERS = {
    "admin":    "password123",
    "user1":    "securepass",
    "testuser": "mypassword",
}


@app.post("/login", response_model=LoginResponse, tags=["Auth"])
async def login(payload: LoginRequest, request: Request):
    """
    Endpoint login yang dilindungi middleware ANN.

    Middleware akan mencegat request ini SEBELUM handler ini dieksekusi.
    Jika IP diblokir, handler ini tidak akan pernah dipanggil.
    """
    # Simulasi pengecekan credential
    expected = DEMO_USERS.get(payload.username)

    if expected is None or expected != payload.password:
        # Kembalikan 401 agar middleware bisa menandai sebagai "gagal"
        raise HTTPException(
            status_code=401,
            detail={
                "error": "invalid_credentials",
                "message": "Username atau password salah.",
            },
        )

    # Login berhasil
    # Dalam produksi: generate JWT token yang proper
    fake_token = f"demo_token_{payload.username}_{int(time.time())}"

    return LoginResponse(
        access_token=fake_token,
        message=f"Selamat datang, {payload.username}!",
    )


@app.post("/api/auth/token", tags=["Auth"])
async def get_token(payload: LoginRequest, request: Request):
    """Endpoint token alternatif (format OAuth2-like)."""
    return await login(payload, request)


@app.post("/api/login", tags=["Auth"])
async def api_login(payload: LoginRequest, request: Request):
    """Endpoint login API."""
    return await login(payload, request)


@app.post("/register", tags=["Auth"])
async def register(payload: RegisterRequest):
    """
    Endpoint registrasi.
    Middleware aktif di sini juga untuk mencegah credential stuffing saat registrasi.
    """
    if payload.username in DEMO_USERS:
        raise HTTPException(status_code=409, detail="Username sudah terdaftar.")

    # Simpan ke database (demo)
    DEMO_USERS[payload.username] = payload.password
    return {"message": f"Akun '{payload.username}' berhasil dibuat.", "status": "created"}


# ═══════════════════════════════════════════════════════════════════════════
# Health Check & Monitoring
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["Monitoring"])
async def health_check():
    """
    Status kesehatan sistem secara keseluruhan.
    Berguna untuk load balancer dan monitoring tools (Prometheus, Grafana).
    """
    redis_ok = redis_store.is_available
    model_ok = inference_engine._model_loaded
    inf_stats = inference_engine.get_stats()

    # Status tier keamanan
    if redis_ok and model_ok:
        security_mode = "FULL_AI"
        status = "healthy"
    elif redis_ok and not model_ok:
        security_mode = "HEURISTIC_FALLBACK"
        status = "degraded"
    else:
        security_mode = "RATE_LIMIT_ONLY"
        status = "degraded"

    return {
        "status": status,
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "security_mode": security_mode,
        "components": {
            "redis": "up" if redis_ok else "down",
            "ann_model": "loaded" if model_ok else "not_loaded",
        },
        "inference_stats": inf_stats,
        "thresholds": {
            "pass":      f"< {settings.THRESHOLD_PASS}",
            "challenge": f"{settings.THRESHOLD_PASS} – {settings.THRESHOLD_CHALLENGE}",
            "block":     f"> {settings.THRESHOLD_CHALLENGE}",
        },
        "protected_paths": settings.PROTECTED_PATHS,
        "timestamp": time.time(),
    }


@app.get("/health/inference", tags=["Monitoring"])
async def inference_health():
    """Detail statistik performa model ANN."""
    return inference_engine.get_stats()


# ═══════════════════════════════════════════════════════════════════════════
# Admin Endpoints
# ═══════════════════════════════════════════════════════════════════════════

ADMIN_SECRET = "admin-secret-key-GANTI-DI-PRODUKSI"


def verify_admin(admin_key: str):
    if admin_key != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Akses admin ditolak.")


@app.post("/admin/unblock", tags=["Admin"])
async def admin_unblock_ip(payload: AdminUnblockRequest):
    """Bebaskan IP dari blokir secara manual."""
    verify_admin(payload.admin_key)
    await redis_store.unblock_ip(payload.ip)
    return {"message": f"IP {payload.ip} berhasil dibebaskan.", "ip": payload.ip}


@app.get("/admin/status/{ip}", tags=["Admin"])
async def admin_check_ip(ip: str, admin_key: str):
    """Cek status blokir sebuah IP."""
    verify_admin(admin_key)
    is_blocked = await redis_store.is_blocked(ip)
    ttl = await redis_store.get_block_ttl(ip) if is_blocked else 0
    records = await redis_store.get_records_in_window(ip, 300)

    return {
        "ip": ip,
        "is_blocked": is_blocked,
        "block_ttl_seconds": ttl,
        "recent_requests_5min": len(records),
        "failed_requests": sum(1 for r in records if not r.success),
        "unique_usernames": len({r.username for r in records}),
    }


@app.delete("/admin/block/{ip}", tags=["Admin"])
async def admin_force_block(ip: str, admin_key: str, duration: int = 900):
    """Blokir IP secara paksa (untuk tindakan manual)."""
    verify_admin(admin_key)
    await redis_store.block_ip(ip, duration)
    return {"message": f"IP {ip} diblokir selama {duration} detik.", "ip": ip}


# ═══════════════════════════════════════════════════════════════════════════
# Root
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Root"])
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


# ── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True,
    )
