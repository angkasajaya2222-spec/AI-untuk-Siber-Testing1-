"""
core/redis_store.py — Redis Sliding Window Store
─────────────────────────────────────────────────
Mendukung dua mode:
  1. Redis asli  : jika Redis server berjalan (produksi)
  2. fakeredis   : Redis tiruan in-memory (development/testing)
                   install dengan: py -3.12 -m pip install fakeredis

Analogi: Redis sebagai "buku catatan" per IP.
fakeredis = buku catatan yang ada di RAM, tidak perlu server terpisah.
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from typing import Optional

# Import Redis — coba yang asli dulu, lalu fakeredis
try:
    import redis.asyncio as aioredis
    _REAL_REDIS = True
except ImportError:
    _REAL_REDIS = False

try:
    import fakeredis.aioredis as fakeredis_async
    _FAKEREDIS_AVAILABLE = True
except ImportError:
    _FAKEREDIS_AVAILABLE = False

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class RequestRecord:
    """Satu catatan request HTTP yang masuk."""
    timestamp: float
    username: str
    success: bool
    user_agent: str
    accept_lang: str
    has_session_cookie: bool
    endpoint: str


class RedisStore:
    """
    Manajemen state request per-IP.
    Otomatis memilih antara Redis asli atau fakeredis.
    """

    def __init__(self):
        self._client = None
        self._available: bool = True

    async def _get_client(self):
        """
        Lazy connection — coba Redis asli dulu, fallback ke fakeredis.
        fakeredis = Redis tiruan yang berjalan di memory Python,
        tanpa perlu install Redis server terpisah.
        """
        if self._client is None:

            # ── Coba Redis asli ───────────────────────────────────────────
            if _REAL_REDIS:
                try:
                    client = aioredis.Redis(
                        host=settings.REDIS_HOST,
                        port=settings.REDIS_PORT,
                        db=settings.REDIS_DB,
                        password=settings.REDIS_PASSWORD or None,
                        socket_timeout=settings.REDIS_TIMEOUT,
                        socket_connect_timeout=settings.REDIS_TIMEOUT,
                        decode_responses=True,
                    )
                    await client.ping()
                    self._client = client
                    self._available = True
                    logger.info("[Redis] ✅ Terhubung ke Redis server.")
                    return self._client
                except Exception as e:
                    logger.warning(f"[Redis] Redis server tidak tersedia: {e}")

            # ── Fallback ke fakeredis ─────────────────────────────────────
            if _FAKEREDIS_AVAILABLE:
                try:
                    self._client = fakeredis_async.FakeRedis(decode_responses=True)
                    await self._client.ping()
                    self._available = True
                    logger.info(
                        "[Redis] ⚡ Menggunakan fakeredis (mode development). "
                        "Semua fitur ANN aktif. Data tidak persisten antar restart."
                    )
                    return self._client
                except Exception as e:
                    logger.warning(f"[Redis] fakeredis gagal: {e}")

            # ── Kedua gagal ───────────────────────────────────────────────
            logger.warning(
                "[Redis] ❌ Tidak ada Redis. Install fakeredis:\n"
                "  py -3.12 -m pip install fakeredis"
            )
            self._available = False

        return self._client

    @property
    def is_available(self) -> bool:
        return self._available

    async def record_request(
        self,
        ip: str,
        username: str,
        success: bool,
        user_agent: str = "",
        accept_lang: str = "",
        has_session_cookie: bool = False,
        endpoint: str = "/login",
    ) -> None:
        client = await self._get_client()
        if client is None:
            return

        record = RequestRecord(
            timestamp=time.time(),
            username=username,
            success=success,
            user_agent=user_agent,
            accept_lang=accept_lang,
            has_session_cookie=has_session_cookie,
            endpoint=endpoint,
        )

        key = f"bf:{ip}"
        try:
            async with client.pipeline(transaction=True) as pipe:
                await pipe.zadd(key, {json.dumps(asdict(record)): record.timestamp})
                cutoff = time.time() - settings.WINDOW_15MIN
                await pipe.zremrangebyscore(key, "-inf", cutoff)
                await pipe.zremrangebyrank(key, 0, -(settings.MAX_HISTORY_SIZE + 1))
                await pipe.expire(key, settings.WINDOW_15MIN + 300)
                await pipe.execute()
        except Exception as e:
            logger.error(f"[Redis] Gagal catat request IP {ip}: {e}")

    async def get_records_in_window(
        self, ip: str, window_seconds: int
    ) -> list[RequestRecord]:
        client = await self._get_client()
        if client is None:
            return []

        key = f"bf:{ip}"
        cutoff = time.time() - window_seconds
        try:
            raw_records = await client.zrangebyscore(key, cutoff, "+inf")
            records = []
            for r in raw_records:
                try:
                    data = json.loads(r)
                    records.append(RequestRecord(**data))
                except (json.JSONDecodeError, TypeError):
                    continue
            return records
        except Exception as e:
            logger.error(f"[Redis] Gagal ambil record IP {ip}: {e}")
            return []

    async def is_blocked(self, ip: str) -> bool:
        client = await self._get_client()
        if client is None:
            return False
        try:
            return bool(await client.exists(f"block:{ip}"))
        except Exception:
            return False

    async def block_ip(self, ip: str, duration: int = None) -> None:
        client = await self._get_client()
        if client is None:
            return
        duration = duration or settings.BLOCK_DURATION_S
        try:
            await client.setex(f"block:{ip}", duration, "1")
            logger.warning(f"[BLOCK] IP {ip} diblokir {duration} detik.")
        except Exception as e:
            logger.error(f"[Redis] Gagal blokir IP {ip}: {e}")

    async def get_block_ttl(self, ip: str) -> int:
        client = await self._get_client()
        if client is None:
            return 0
        try:
            ttl = await client.ttl(f"block:{ip}")
            return max(0, ttl)
        except Exception:
            return 0

    async def unblock_ip(self, ip: str) -> None:
        client = await self._get_client()
        if client is None:
            return
        try:
            await client.delete(f"block:{ip}")
            logger.info(f"[UNBLOCK] IP {ip} dibebaskan.")
        except Exception as e:
            logger.error(f"[Redis] Gagal unblock IP {ip}: {e}")

    async def get_fallback_count(self, ip: str) -> int:
        key = f"_mem_fallback_{ip}"
        if not hasattr(self, "_mem_store"):
            self._mem_store: dict = {}
        now = time.time()
        window_start = now - settings.FALLBACK_WINDOW_S
        entries = self._mem_store.get(key, [])
        entries = [t for t in entries if t > window_start]
        entries.append(now)
        self._mem_store[key] = entries
        return len(entries)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            logger.info("[Redis] Koneksi ditutup.")


# Singleton global
redis_store = RedisStore()
