"""Async latency/throughput load test for the emotion API.

Hammers ``POST /predict`` with a fixed concurrency and reports a latency
distribution (p50/p95/p99) and sustained throughput (req/s), in the same spirit
as the python-perf-optimizer benchmarker. Designed to run against the offline
stub so the numbers are reproducible without a model download.

It can drive an already-running server (``--url``) or, by default, spin up an
in-process ASGI server on an ephemeral port, benchmark it, and tear it down --
so ``python scripts/loadtest.py`` works from a clean checkout with no setup.

Usage:
    python scripts/loadtest.py                       # self-hosted, offline
    python scripts/loadtest.py --requests 5000 --concurrency 64
    python scripts/loadtest.py --url http://localhost:8000   # external server
    python scripts/loadtest.py --markdown             # emit the README table

Output (default) is a human-readable summary; ``--markdown`` emits the exact
benchmark table used in the README, and ``--json`` emits machine-readable stats.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import socket
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

# Make the project importable when run as a plain script (no install needed).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging

import httpx

# httpx logs one INFO line per request; under load that I/O dominates the
# measurement. Silence it (and uvicorn) so the benchmark times the server.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

SAMPLE_TEXTS = [
    "i can't stop smiling, today went better than i ever hoped",
    "i feel so let down after everything we had planned fell apart",
    "my hands are shaking, i really don't think i can walk in there",
    "how dare they take credit for the work i did all weekend",
    "i didn't expect a package on my doorstep and now i'm grinning",
    "i just feel really tender and grateful for the people around me",
    "this is fine i guess, nothing special happened today",
    "wow i genuinely did not see that coming at all",
]


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def _ephemeral_server(host: str = "127.0.0.1"):
    """Run the app in a background uvicorn server for the duration of the block."""
    import os

    import uvicorn

    os.environ.setdefault("OFFLINE", "1")
    from app.main import create_app

    port = _free_port()
    config = uvicorn.Config(create_app(), host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    # Wait for readiness.
    base = f"http://{host}:{port}"
    deadline = time.time() + 20
    with httpx.Client(timeout=2.0) as c:
        while time.time() < deadline:
            try:
                if c.get(f"{base}/healthz").status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.05)
        else:
            raise RuntimeError("server did not become ready in time")
    try:
        yield base
    finally:
        server.should_exit = True
        thread.join(timeout=10)


async def _worker(
    client: httpx.AsyncClient,
    url: str,
    work: "asyncio.Queue[int]",
    latencies: List[float],
    errors: List[int],
) -> None:
    while True:
        try:
            i = work.get_nowait()
        except asyncio.QueueEmpty:
            return
        payload = {"text": SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]}
        t0 = time.perf_counter()
        try:
            r = await client.post(url, json=payload)
            dt = (time.perf_counter() - t0) * 1000.0  # ms
            if r.status_code == 200:
                latencies.append(dt)
            else:
                errors.append(r.status_code)
        except httpx.HTTPError:
            errors.append(-1)
        finally:
            work.task_done()


async def run_benchmark(
    base_url: str,
    total_requests: int,
    concurrency: int,
    warmup: int,
) -> Dict[str, float]:
    endpoint = base_url.rstrip("/") + "/predict"
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(timeout=30.0, limits=limits) as client:
        # Warm-up (not measured): primes connections and the batcher worker.
        for i in range(warmup):
            with contextlib.suppress(httpx.HTTPError):
                await client.post(endpoint, json={"text": SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]})

        work: "asyncio.Queue[int]" = asyncio.Queue()
        for i in range(total_requests):
            work.put_nowait(i)
        latencies: List[float] = []
        errors: List[int] = []

        start = time.perf_counter()
        workers = [
            asyncio.create_task(_worker(client, endpoint, work, latencies, errors))
            for _ in range(concurrency)
        ]
        await asyncio.gather(*workers)
        wall = time.perf_counter() - start

    ok = len(latencies)
    return {
        "requests": total_requests,
        "concurrency": concurrency,
        "ok": ok,
        "errors": len(errors),
        "wall_s": wall,
        "throughput_rps": ok / wall if wall > 0 else float("nan"),
        "mean_ms": statistics.fmean(latencies) if latencies else float("nan"),
        "p50_ms": _percentile(latencies, 50),
        "p95_ms": _percentile(latencies, 95),
        "p99_ms": _percentile(latencies, 99),
        "min_ms": min(latencies) if latencies else float("nan"),
        "max_ms": max(latencies) if latencies else float("nan"),
    }


def _print_human(s: Dict[str, float]) -> None:
    print("\nEmotion API load test (offline stub)")
    print("=" * 44)
    print(f"  requests       : {s['requests']}")
    print(f"  concurrency    : {s['concurrency']}")
    print(f"  ok / errors    : {s['ok']} / {s['errors']}")
    print(f"  wall time      : {s['wall_s']:.3f} s")
    print(f"  throughput     : {s['throughput_rps']:.1f} req/s")
    print(f"  latency mean   : {s['mean_ms']:.3f} ms")
    print(f"  latency p50    : {s['p50_ms']:.3f} ms")
    print(f"  latency p95    : {s['p95_ms']:.3f} ms")
    print(f"  latency p99    : {s['p99_ms']:.3f} ms")
    print(f"  latency range  : {s['min_ms']:.3f} – {s['max_ms']:.3f} ms")


def _print_markdown(s: Dict[str, float]) -> None:
    print("\n| concurrency | throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) |")
    print("|---|---|---|---|---|")
    print(
        f"| {s['concurrency']} | {s['throughput_rps']:.0f} | "
        f"{s['p50_ms']:.2f} | {s['p95_ms']:.2f} | {s['p99_ms']:.2f} |"
    )


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Latency/throughput load test for the emotion API.")
    ap.add_argument("--url", default=None, help="Base URL of a running server. If omitted, an in-process server is started.")
    ap.add_argument("--requests", type=int, default=2000, help="Total requests to send (default 2000).")
    ap.add_argument("--concurrency", type=int, default=32, help="Concurrent in-flight requests (default 32).")
    ap.add_argument("--warmup", type=int, default=50, help="Unmeasured warm-up requests (default 50).")
    ap.add_argument("--markdown", action="store_true", help="Emit the README benchmark table.")
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON stats.")
    args = ap.parse_args(argv)

    def _bench(base: str) -> Dict[str, float]:
        return asyncio.run(run_benchmark(base, args.requests, args.concurrency, args.warmup))

    if args.url:
        stats = _bench(args.url)
    else:
        with _ephemeral_server() as base:
            stats = _bench(base)

    if args.json:
        print(json.dumps(stats, indent=2))
    elif args.markdown:
        _print_markdown(stats)
    else:
        _print_human(stats)


if __name__ == "__main__":
    main()
