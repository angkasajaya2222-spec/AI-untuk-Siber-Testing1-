"""
tests/test_middleware.py — Test Suite Lengkap
──────────────────────────────────────────────
Menguji seluruh pipeline dari ekstraksi fitur hingga
respons middleware dengan skenario realistis:

  1. Unit test: Feature Extractor
  2. Unit test: ANN Model (forward pass, bentuk output)
  3. Unit test: Inference Engine (scoring, fallback)
  4. Integration test: Middleware (PASS / CHALLENGE / BLOCK)
  5. Load test: latensi end-to-end

Cara jalankan:
  pytest tests/test_middleware.py -v
  pytest tests/test_middleware.py -v --tb=short -x
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

# ─────────────────────────────────────────────
# Fixtures dan Helpers
# ─────────────────────────────────────────────

@pytest.fixture(scope="session")
def event_loop():
    """Event loop untuk seluruh sesi test."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def make_feature_vector(**overrides):
    """Buat ExtractedFeatures dengan nilai default, bisa di-override."""
    from core.feature_extractor import ExtractedFeatures

    defaults = dict(
        req_count_1min=1,
        req_count_5min=3,
        failure_rate_1min=0.0,
        failure_rate_5min=0.0,
        unique_usernames_1m=1,
        unique_usernames_5m=1,
        interval_mean_ms=5000.0,
        interval_std_ms=1500.0,
        ua_entropy=5.2,
        ua_is_generic=False,
        has_session_cookie=True,
        accept_lang_missing=False,
        time_of_day_norm=0.5,
        burst_score=0.0,
    )
    defaults.update(overrides)
    return ExtractedFeatures(**defaults)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Unit Test: Feature Extractor
# ═══════════════════════════════════════════════════════════════════════════

class TestFeatureExtractor:

    def test_normal_features_shape(self):
        """Vektor fitur harus memiliki 14 dimensi."""
        feat = make_feature_vector()
        arr = feat.to_numpy()
        assert arr.shape == (14,), f"Harap shape (14,), dapat {arr.shape}"

    def test_features_normalized_range(self):
        """Semua nilai fitur harus dalam rentang [0.0, 1.0]."""
        feat = make_feature_vector()
        arr = feat.to_numpy()
        assert np.all(arr >= 0.0), f"Ada nilai negatif: {arr[arr < 0]}"
        assert np.all(arr <= 1.0), f"Ada nilai > 1: {arr[arr > 1]}"

    def test_high_request_count_clamped(self):
        """Request count sangat tinggi harus di-clamp ke 1.0."""
        feat = make_feature_vector(req_count_1min=9999, req_count_5min=9999)
        arr = feat.to_numpy()
        assert arr[0] == pytest.approx(1.0, abs=0.01)
        assert arr[1] == pytest.approx(1.0, abs=0.01)

    def test_full_failure_rate(self):
        """Failure rate 100% harus tercermin di fitur."""
        feat = make_feature_vector(failure_rate_1min=1.0, failure_rate_5min=1.0)
        arr = feat.to_numpy()
        assert arr[2] == pytest.approx(1.0)
        assert arr[3] == pytest.approx(1.0)

    def test_no_session_cookie_flag(self):
        """Tidak ada cookie sesi harus menghasilkan fitur = 1.0."""
        feat_no_cookie  = make_feature_vector(has_session_cookie=False)
        feat_has_cookie = make_feature_vector(has_session_cookie=True)
        # Fitur [10] = float(not has_session_cookie)
        assert feat_no_cookie.to_numpy()[10] == 1.0
        assert feat_has_cookie.to_numpy()[10] == 0.0

    def test_generic_ua_flag(self):
        """User-Agent generik harus menghasilkan fitur = 1.0."""
        feat_generic = make_feature_vector(ua_is_generic=True)
        feat_normal  = make_feature_vector(ua_is_generic=False)
        assert feat_generic.to_numpy()[9] == 1.0
        assert feat_normal.to_numpy()[9] == 0.0

    def test_shannon_entropy_calculation(self):
        """Test perhitungan Shannon entropy."""
        from core.feature_extractor import _shannon_entropy

        assert _shannon_entropy("") == 0.0
        assert _shannon_entropy("aaaa") < _shannon_entropy("abcd")
        assert _shannon_entropy("abc") > 0.0

    def test_suspicious_ua_detection(self):
        """Test deteksi User-Agent mencurigakan."""
        from core.feature_extractor import _is_generic_ua

        # UA mencurigakan
        assert _is_generic_ua("python-requests/2.31.0") is True
        assert _is_generic_ua("curl/7.88.1") is True
        assert _is_generic_ua("Hydra/v9.5") is True
        assert _is_generic_ua("") is True
        assert _is_generic_ua("ab") is True  # terlalu pendek

        # UA normal
        assert _is_generic_ua(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        ) is False


# ═══════════════════════════════════════════════════════════════════════════
# 2. Unit Test: Model ANN
# ═══════════════════════════════════════════════════════════════════════════

class TestANNModel:

    def test_model_output_shape(self):
        """Output model harus berupa skor skalar [0.0, 1.0]."""
        from models.ann_model import BruteForceANN

        model = BruteForceANN(input_dim=14)
        model.eval()

        x = torch.rand(1, 14)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (1, 1), f"Expected (1,1), got {out.shape}"
        assert 0.0 <= out.item() <= 1.0

    def test_model_batch_inference(self):
        """Model harus bisa memproses batch input."""
        from models.ann_model import BruteForceANN

        model = BruteForceANN(input_dim=14)
        model.eval()

        batch = torch.rand(32, 14)
        with torch.no_grad():
            out = model(batch)

        assert out.shape == (32, 1)
        assert torch.all(out >= 0.0) and torch.all(out <= 1.0)

    def test_model_sigmoid_output(self):
        """Output harus selalu dalam [0, 1] untuk semua input ekstrem."""
        from models.ann_model import BruteForceANN

        model = BruteForceANN(input_dim=14)
        model.eval()

        with torch.no_grad():
            out_zeros = model(torch.zeros(1, 14))
            out_ones  = model(torch.ones(1, 14))
            out_large = model(torch.full((1, 14), 1000.0))
            out_neg   = model(torch.full((1, 14), -1000.0))

        for out in [out_zeros, out_ones, out_large, out_neg]:
            assert 0.0 <= out.item() <= 1.0, f"Output di luar [0,1]: {out.item()}"

    def test_model_save_load(self, tmp_path):
        """Model harus bisa disimpan dan dimuat ulang dengan bobot yang sama."""
        from models.ann_model import BruteForceANN, save_model, load_model

        model = BruteForceANN(input_dim=14)
        model.eval()

        x = torch.rand(1, 14)
        with torch.no_grad():
            original_out = model(x).item()

        path = tmp_path / "test_model.pt"
        save_model(model, path)

        loaded = load_model(path, input_dim=14)
        loaded.eval()
        with torch.no_grad():
            loaded_out = loaded(x).item()

        assert abs(original_out - loaded_out) < 1e-6, "Bobot model berubah setelah save/load!"

    def test_inference_latency(self):
        """Inferensi tunggal harus selesai dalam 50ms."""
        from models.ann_model import BruteForceANN

        model = BruteForceANN(input_dim=14)
        model.eval()

        x = torch.rand(1, 14)

        # Warm-up
        with torch.no_grad():
            _ = model(x)

        # Ukur latensi 100 inferensi
        latencies = []
        for _ in range(100):
            t0 = time.perf_counter()
            with torch.no_grad():
                model(x)
            latencies.append((time.perf_counter() - t0) * 1000)

        avg_ms = sum(latencies) / len(latencies)
        p95_ms = sorted(latencies)[95]

        print(f"\n  Latensi rata-rata: {avg_ms:.2f}ms, P95: {p95_ms:.2f}ms")
        assert avg_ms < 50.0, f"Latensi rata-rata terlalu tinggi: {avg_ms:.2f}ms"
        assert p95_ms < 50.0, f"Latensi P95 terlalu tinggi: {p95_ms:.2f}ms"


# ═══════════════════════════════════════════════════════════════════════════
# 3. Unit Test: Inference Engine (dengan mock)
# ═══════════════════════════════════════════════════════════════════════════

class TestInferenceEngine:

    @pytest.mark.asyncio
    async def test_predict_returns_float(self):
        """predict() harus mengembalikan float dalam [0.0, 1.0]."""
        from core.inference_engine import InferenceEngine

        engine = InferenceEngine()
        feat = make_feature_vector()

        # Tanpa model dimuat → gunakan fallback heuristic
        score = await engine.predict(feat)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_fallback_heuristic_normal_traffic(self):
        """Traffic normal harus menghasilkan skor rendah di heuristic."""
        from core.inference_engine import InferenceEngine

        engine = InferenceEngine()
        feat = make_feature_vector(
            req_count_1min=1,
            failure_rate_1min=0.0,
            unique_usernames_1m=1,
            ua_is_generic=False,
        )
        score = engine._fallback_heuristic(feat)
        assert score < 0.50, f"Normal traffic harusnya skor < 0.5, dapat {score}"

    @pytest.mark.asyncio
    async def test_fallback_heuristic_brute_force(self):
        """Traffic brute-force harus menghasilkan skor tinggi di heuristic."""
        from core.inference_engine import InferenceEngine

        engine = InferenceEngine()
        feat = make_feature_vector(
            req_count_1min=60,
            failure_rate_1min=0.95,
            unique_usernames_1m=30,
            ua_is_generic=True,
            interval_std_ms=10.0,  # bot: std sangat rendah
        )
        score = engine._fallback_heuristic(feat)
        assert score > 0.60, f"Brute-force harusnya skor > 0.6, dapat {score}"


# ═══════════════════════════════════════════════════════════════════════════
# 4. Integration Test: Middleware via TestClient
# ═══════════════════════════════════════════════════════════════════════════

class TestMiddlewareIntegration:
    """
    Test integrasi menggunakan FastAPI TestClient.
    Redis dan ANN di-mock agar test berjalan tanpa infrastruktur.
    """

    @pytest.fixture(autouse=True)
    def setup_mocks(self, monkeypatch):
        """
        Mock Redis dan inference engine untuk isolasi test.
        """
        # Mock Redis store
        mock_redis = AsyncMock()
        mock_redis.is_available = True
        mock_redis.is_blocked = AsyncMock(return_value=False)
        mock_redis.get_block_ttl = AsyncMock(return_value=0)
        mock_redis.record_request = AsyncMock()
        mock_redis.get_records_in_window = AsyncMock(return_value=[])
        mock_redis.block_ip = AsyncMock()
        mock_redis.get_fallback_count = AsyncMock(return_value=1)

        monkeypatch.setattr("middleware.ann_middleware.redis_store", mock_redis)
        monkeypatch.setattr("core.feature_extractor.redis_store", mock_redis)
        self.mock_redis = mock_redis

    @pytest.fixture
    def client_with_score(self, setup_mocks, monkeypatch):
        """Factory: buat test client dengan skor ANN tertentu."""
        def _make_client(score: float):
            mock_engine = AsyncMock()
            mock_engine._model_loaded = True
            mock_engine.predict = AsyncMock(return_value=score)

            monkeypatch.setattr("middleware.ann_middleware.inference_engine", mock_engine)
            monkeypatch.setattr("core.inference_engine.inference_engine", mock_engine)

            from main import app
            return TestClient(app, raise_server_exceptions=False)

        return _make_client

    def test_health_endpoint(self, setup_mocks, monkeypatch):
        """Health endpoint harus selalu merespons 200."""
        mock_engine = MagicMock()
        mock_engine._model_loaded = True
        mock_engine.get_stats = MagicMock(return_value={
            "model_loaded": True,
            "total_inferences": 0,
            "failed_inferences": 0,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
        })
        monkeypatch.setattr("main.inference_engine", mock_engine)

        from main import app
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "security_mode" in data

    def test_pass_low_threat_score(self, client_with_score):
        """Skor < 0.40 → request harus lolos (200)."""
        client = client_with_score(0.15)
        resp = client.post("/login", json={"username": "admin", "password": "password123"})
        # Skor rendah → middleware membiarkan request masuk ke handler
        assert resp.status_code in (200, 401)  # 401 = credential salah, tapi middleware lolos
        assert resp.status_code != 429  # Bukan "too many requests"

    def test_challenge_medium_threat_score(self, client_with_score):
        """Skor 0.40–0.70 → response harus mengandung header CAPTCHA."""
        client = client_with_score(0.55)
        resp = client.post("/login", json={"username": "user1", "password": "wrongpass"})
        assert resp.status_code != 429
        assert "X-Challenge-Required" in resp.headers
        assert resp.headers["X-Challenge-Required"] == "recaptcha"

    def test_block_high_threat_score(self, client_with_score):
        """Skor > 0.70 → request harus diblokir (429)."""
        client = client_with_score(0.85)
        resp = client.post("/login", json={"username": "victim", "password": "guess"})
        assert resp.status_code == 429
        data = resp.json()
        assert "retry_after" in data
        assert data["error"] == "too_many_requests"

    def test_blocked_ip_returns_429(self, setup_mocks, monkeypatch):
        """IP yang sudah diblokir harus langsung mendapat 429."""
        self.mock_redis.is_blocked = AsyncMock(return_value=True)
        self.mock_redis.get_block_ttl = AsyncMock(return_value=600)

        from main import app
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/login", json={"username": "anyuser", "password": "anypass"})
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers

    def test_unprotected_path_bypasses_middleware(self, client_with_score):
        """Path yang tidak dilindungi tidak boleh dicegat middleware."""
        client = client_with_score(0.99)  # Skor sangat tinggi sekalipun
        resp = client.get("/health")
        assert resp.status_code == 200  # Bukan 429

    def test_retry_after_header_present(self, client_with_score):
        """Response blokir harus menyertakan header Retry-After."""
        client = client_with_score(0.90)
        resp = client.post("/login", json={"username": "x", "password": "y"})
        if resp.status_code == 429:
            assert "Retry-After" in resp.headers
            assert int(resp.headers["Retry-After"]) > 0

    def test_block_reason_header(self, client_with_score):
        """Response blokir harus menyertakan X-Block-Reason."""
        client = client_with_score(0.90)
        resp = client.post("/login", json={"username": "x", "password": "y"})
        if resp.status_code == 429:
            assert resp.headers.get("X-Block-Reason") == "brute_force_detected"


# ═══════════════════════════════════════════════════════════════════════════
# 5. Scenario Test: Simulasi Serangan
# ═══════════════════════════════════════════════════════════════════════════

class TestAttackScenarios:
    """
    Skenario simulasi serangan end-to-end.
    Test ini memvalidasi logika bisnis sistem secara holistik.
    """

    def test_decision_thresholds(self):
        """Validasi bahwa threshold menghasilkan aksi yang benar."""
        from middleware.ann_middleware import ANNBruteForceMiddleware
        from core.security_logger import ThreatAction

        assert ANNBruteForceMiddleware._decide_action(0.10) == ThreatAction.PASS
        assert ANNBruteForceMiddleware._decide_action(0.39) == ThreatAction.PASS
        assert ANNBruteForceMiddleware._decide_action(0.40) == ThreatAction.CHALLENGE
        assert ANNBruteForceMiddleware._decide_action(0.55) == ThreatAction.CHALLENGE
        assert ANNBruteForceMiddleware._decide_action(0.69) == ThreatAction.CHALLENGE
        assert ANNBruteForceMiddleware._decide_action(0.70) == ThreatAction.BLOCK
        assert ANNBruteForceMiddleware._decide_action(0.99) == ThreatAction.BLOCK

    def test_credential_stuffing_features(self):
        """
        Credential stuffing: satu IP, banyak username berbeda.
        Heuristik harus memberikan skor tinggi.
        """
        from core.inference_engine import InferenceEngine

        engine = InferenceEngine()
        feat = make_feature_vector(
            req_count_1min=40,
            failure_rate_1min=0.90,
            unique_usernames_1m=35,    # ← ciri khas credential stuffing
            ua_is_generic=True,
            interval_std_ms=5.0,       # sangat konsisten (bot)
        )
        score = engine._fallback_heuristic(feat)
        assert score > 0.65, (
            f"Credential stuffing harusnya terdeteksi (score>0.65), "
            f"dapat {score:.3f}"
        )

    def test_normal_user_low_score(self):
        """Pengguna normal tidak boleh terdeteksi sebagai ancaman."""
        from core.inference_engine import InferenceEngine

        engine = InferenceEngine()
        feat = make_feature_vector(
            req_count_1min=1,
            req_count_5min=2,
            failure_rate_1min=0.0,
            unique_usernames_1m=1,
            interval_mean_ms=8000.0,   # manusia: interval bervariasi
            interval_std_ms=3000.0,    # std tinggi = manusia
            ua_is_generic=False,
            has_session_cookie=True,
        )
        score = engine._fallback_heuristic(feat)
        assert score < 0.45, (
            f"Pengguna normal seharusnya lolos (score<0.45), "
            f"dapat {score:.3f}"
        )

    @pytest.mark.asyncio
    async def test_protected_path_detection(self):
        """Path yang dilindungi harus terdeteksi dengan benar."""
        from middleware.ann_middleware import ANNBruteForceMiddleware

        protected = ["/login", "/auth/token", "/api/login", "/api/auth"]
        unprotected = ["/", "/health", "/docs", "/static/style.css", "/about"]

        for path in protected:
            assert ANNBruteForceMiddleware._is_protected(path), \
                f"Path {path} seharusnya dilindungi!"

        for path in unprotected:
            assert not ANNBruteForceMiddleware._is_protected(path), \
                f"Path {path} tidak seharusnya dilindungi!"


# ─────────────────────────────────────────────
# Jalankan langsung (tanpa pytest)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
