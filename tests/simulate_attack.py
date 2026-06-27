"""
tests/simulate_attack.py — Simulator Serangan Brute-Force
───────────────────────────────────────────────────────────
Script untuk menguji sistem secara live dengan mensimulasikan
berbagai skenario serangan. Jalankan ini saat server sedang aktif.

Cara pakai:
  # Pastikan server berjalan: uvicorn main:app --port 8000
  python tests/simulate_attack.py --mode brute_force
  python tests/simulate_attack.py --mode credential_stuffing
  python tests/simulate_attack.py --mode normal_user
  python tests/simulate_attack.py --mode all
"""

import argparse
import asyncio
import json
import random
import time
from dataclasses import dataclass
from typing import Literal

import httpx
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()

BASE_URL = "http://localhost:8000"
LOGIN_URL = f"{BASE_URL}/login"


@dataclass
class SimResult:
    attempt: int
    username: str
    password: str
    status_code: int
    latency_ms: float
    blocked: bool
    challenge: bool
    response_body: dict


# ═══════════════════════════════════════════════════════════════════════════
# Skenario Serangan
# ═══════════════════════════════════════════════════════════════════════════

async def simulate_brute_force(
    target_username: str = "admin",
    total_attempts: int = 30,
    delay_ms: float = 200,
) -> list[SimResult]:
    """
    Brute Force Klasik: satu username, banyak password.
    Ini adalah serangan paling umum — attacker mencoba satu akun
    dengan ribuan kombinasi password.
    """
    console.print("\n[bold red]🔴 SIMULASI: Brute Force Attack[/bold red]")
    console.print(f"Target: {target_username}, Percobaan: {total_attempts}")

    results = []
    passwords = [
        "password", "123456", "admin", "letmein", "qwerty",
        "password1", "abc123", "monkey", "1234567890", "dragon",
        "master", "123123", "superman", "batman", "princess",
    ] * 3  # ulangi untuk mencapai jumlah yang diinginkan

    async with httpx.AsyncClient(timeout=10.0) as client:
        for i in range(total_attempts):
            pwd = passwords[i % len(passwords)]
            t0 = time.perf_counter()
            try:
                resp = await client.post(
                    LOGIN_URL,
                    json={"username": target_username, "password": pwd},
                    headers={
                        "User-Agent": "python-requests/2.28.0",  # UA mencurigakan
                        "X-Forwarded-For": "192.168.1.100",       # IP tetap
                    },
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}

                result = SimResult(
                    attempt=i+1,
                    username=target_username,
                    password=pwd,
                    status_code=resp.status_code,
                    latency_ms=latency_ms,
                    blocked=resp.status_code == 429,
                    challenge="X-Challenge-Required" in resp.headers,
                    response_body=body,
                )
                results.append(result)

                # Tampilkan status
                icon = "🔴" if result.blocked else ("🟡" if result.challenge else "🟢")
                console.print(
                    f"  [{i+1:3d}] {icon} {resp.status_code} "
                    f"({latency_ms:5.0f}ms) pwd={pwd[:15]:<15}"
                )

                if result.blocked:
                    console.print(f"       [bold red]⛔ IP DIBLOKIR! Retry-After: {resp.headers.get('Retry-After', '?')}s[/bold red]")

            except Exception as e:
                console.print(f"  [{i+1:3d}] ❌ Error: {e}")

            await asyncio.sleep(delay_ms / 1000.0)

    return results


async def simulate_credential_stuffing(
    total_attempts: int = 25,
    delay_ms: float = 300,
) -> list[SimResult]:
    """
    Credential Stuffing: banyak username berbeda dari satu IP.
    Attacker menggunakan daftar credential yang bocor dari breach lain.
    """
    console.print("\n[bold yellow]🟡 SIMULASI: Credential Stuffing Attack[/bold yellow]")
    console.print(f"Total: {total_attempts} kombinasi unik")

    # Simulasi daftar credential bocor
    leaked_credentials = [
        ("john.doe@gmail.com", "Sunshine2023!"),
        ("jane_smith", "MyDog2022"),
        ("mike.wilson", "Football99"),
        ("sarah_j", "Ilovemusic1"),
        ("robert123", "Qwerty@456"),
        ("emily.clark", "Purple$Rain"),
        ("david_wong", "Tech2024!"),
        ("lisa.anderson", "Summer#789"),
        ("james_taylor", "Winter2023"),
        ("maria.garcia", "Flores123!"),
    ] * 3

    results = []
    async with httpx.AsyncClient(timeout=10.0) as client:
        for i in range(min(total_attempts, len(leaked_credentials))):
            username, password = leaked_credentials[i]
            t0 = time.perf_counter()
            try:
                resp = await client.post(
                    LOGIN_URL,
                    json={"username": username, "password": password},
                    headers={
                        "User-Agent": "Mozilla/5.0 (compatible; automated)",
                        "X-Forwarded-For": "10.0.0.55",  # IP sama → kunci deteksi
                        "Accept-Language": "",
                    },
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}

                result = SimResult(
                    attempt=i+1, username=username, password=password,
                    status_code=resp.status_code, latency_ms=latency_ms,
                    blocked=resp.status_code == 429,
                    challenge="X-Challenge-Required" in resp.headers,
                    response_body=body,
                )
                results.append(result)

                icon = "🔴" if result.blocked else ("🟡" if result.challenge else "🟢")
                console.print(
                    f"  [{i+1:3d}] {icon} {resp.status_code} "
                    f"({latency_ms:5.0f}ms) user={username[:25]:<25}"
                )

                if result.blocked:
                    break

            except Exception as e:
                console.print(f"  [{i+1:3d}] ❌ Error: {e}")

            await asyncio.sleep(delay_ms / 1000.0)

    return results


async def simulate_normal_user(
    username: str = "admin",
    total_sessions: int = 5,
) -> list[SimResult]:
    """
    Pengguna Normal: login sesekali dengan jeda manusiawi.
    Hasilnya harus selalu PASS (tidak terblokir).
    """
    console.print("\n[bold green]🟢 SIMULASI: Pengguna Normal[/bold green]")
    console.print(f"Sesi: {total_sessions}, interval: 2-5 detik")

    results = []
    async with httpx.AsyncClient(timeout=10.0) as client:
        for i in range(total_sessions):
            # Kadang salah password (manusiawi)
            is_correct = random.random() > 0.3
            password = "password123" if is_correct else "wrongpass"

            t0 = time.perf_counter()
            resp = await client.post(
                LOGIN_URL,
                json={"username": username, "password": password},
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    "X-Forwarded-For": "172.16.0.10",
                    "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8",
                    "Cookie": "session_id=abc123xyz",
                },
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            result = SimResult(
                attempt=i+1, username=username, password=password,
                status_code=resp.status_code, latency_ms=latency_ms,
                blocked=resp.status_code == 429,
                challenge="X-Challenge-Required" in resp.headers,
                response_body={},
            )
            results.append(result)

            status_icon = "✅" if resp.status_code == 200 else "❌"
            console.print(
                f"  [{i+1}] {status_icon} {resp.status_code} "
                f"({latency_ms:.0f}ms) {'[bold green]LOGIN SUKSES[/bold green]' if is_correct else 'salah password'}"
            )
            assert not result.blocked, "❌ MASALAH: Pengguna normal terblokir! (False Positive)"

            # Jeda manusiawi
            await asyncio.sleep(random.uniform(2.0, 5.0))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Report Generator
# ═══════════════════════════════════════════════════════════════════════════

def print_summary(results: list[SimResult], scenario_name: str) -> None:
    """Cetak ringkasan hasil simulasi."""
    if not results:
        return

    table = Table(title=f"📊 Ringkasan: {scenario_name}", show_lines=True)
    table.add_column("Metrik", style="cyan")
    table.add_column("Nilai", style="bold white")

    total      = len(results)
    blocked    = sum(1 for r in results if r.blocked)
    challenged = sum(1 for r in results if r.challenge)
    passed     = total - blocked - challenged
    avg_lat    = sum(r.latency_ms for r in results) / total

    first_block = next((r.attempt for r in results if r.blocked), None)

    table.add_row("Total Percobaan",     str(total))
    table.add_row("✅ Diizinkan (PASS)", f"{passed} ({passed/total*100:.1f}%)")
    table.add_row("🟡 Tantangan (CAPTCHA)", f"{challenged} ({challenged/total*100:.1f}%)")
    table.add_row("🔴 Diblokir (BLOCK)",  f"{blocked} ({blocked/total*100:.1f}%)")
    table.add_row("Deteksi Pertama",     f"Percobaan ke-{first_block}" if first_block else "Tidak terdeteksi")
    table.add_row("Latensi Rata-rata",   f"{avg_lat:.0f}ms")

    console.print(table)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

async def main(mode: str):
    console.rule("[bold blue]🛡️  ANN Brute-Force Detection — Simulator[/bold blue]")

    # Cek server aktif
    try:
        async with httpx.AsyncClient() as c:
            resp = await c.get(f"{BASE_URL}/health", timeout=3.0)
            data = resp.json()
            console.print(f"[green]✅ Server aktif: mode={data.get('security_mode', 'unknown')}[/green]")
    except Exception:
        console.print("[red]❌ Server tidak berjalan! Jalankan: uvicorn main:app --port 8000[/red]")
        return

    all_results = {}

    if mode in ("brute_force", "all"):
        r = await simulate_brute_force(total_attempts=25, delay_ms=150)
        all_results["Brute Force"] = r
        print_summary(r, "Brute Force Attack")

    if mode in ("credential_stuffing", "all"):
        await asyncio.sleep(2)  # jeda antar skenario
        r = await simulate_credential_stuffing(total_attempts=20, delay_ms=200)
        all_results["Credential Stuffing"] = r
        print_summary(r, "Credential Stuffing")

    if mode in ("normal_user", "all"):
        await asyncio.sleep(2)
        r = await simulate_normal_user(total_sessions=5)
        all_results["Normal User"] = r
        print_summary(r, "Pengguna Normal")

    console.rule("[bold blue]Simulasi Selesai[/bold blue]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulator serangan brute-force")
    parser.add_argument(
        "--mode",
        choices=["brute_force", "credential_stuffing", "normal_user", "all"],
        default="all",
        help="Mode simulasi yang dijalankan",
    )
    args = parser.parse_args()
    asyncio.run(main(args.mode))
