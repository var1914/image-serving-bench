#!/usr/bin/env python3
"""Make the event-loop-blocking bug visible with numbers.

Fires K image requests CONCURRENTLY at one endpoint, and at the same time pings
/healthz every few ms. Two things to watch:
  1. total wall time for the K requests  -> blocking serializes; offloaded parallelizes
  2. /healthz latency DURING the burst    -> blocking freezes the canary; offloaded doesn't

Usage: .venv/bin/python harness/concurrency_probe.py            # runs both endpoints
"""
from __future__ import annotations
import asyncio, io, statistics, sys, time
import httpx
from PIL import Image
import numpy as np

BASE = "http://127.0.0.1:8099"
K = 16              # concurrent image requests
MP = 6             # test image megapixels (bigger -> longer decode -> starker effect)


def make_image(mp: int) -> bytes:
    side = int((mp * 1e6) ** 0.5)
    xr = np.linspace(0, 255, side, dtype="uint8")
    arr = np.stack([np.broadcast_to(xr, (side, side)),
                    np.broadcast_to(xr[:, None], (side, side)),
                    np.broadcast_to(xr, (side, side)) // 2], axis=-1)
    buf = io.BytesIO(); Image.fromarray(arr, "RGB").save(buf, "JPEG", quality=85)
    return buf.getvalue()


async def wait_ready(timeout=20.0):
    async with httpx.AsyncClient() as c:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            try:
                if (await c.get(BASE + "/healthz", timeout=1)).status_code == 200:
                    return True
            except Exception:
                pass
            await asyncio.sleep(0.15)
    return False


def pct(a, p):
    a = sorted(a); return a[min(len(a) - 1, int(p / 100 * len(a)))] if a else float("nan")


async def run_endpoint(name: str, path: str, body: bytes):
    """Fire K concurrent POSTs while a background task pings /healthz."""
    healthz_lat, stop = [], False

    async def health_pinger(client):
        while not stop:
            t = time.perf_counter()
            try:
                await client.get(BASE + "/healthz", timeout=10)
                healthz_lat.append((time.perf_counter() - t) * 1e3)
            except Exception:
                healthz_lat.append(float("nan"))
            await asyncio.sleep(0.01)

    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=K + 4)) as client:
        pinger = asyncio.create_task(health_pinger(client))
        await asyncio.sleep(0.05)                 # let the pinger establish a baseline
        req_lat = []

        async def one():
            t = time.perf_counter()
            r = await client.post(BASE + path, content=body, timeout=60)
            req_lat.append((time.perf_counter() - t) * 1e3)
            return r.status_code

        wall0 = time.perf_counter()
        codes = await asyncio.gather(*[one() for _ in range(K)])
        wall = (time.perf_counter() - wall0) * 1e3
        stop = True
        await pinger

    ok = sum(c == 200 for c in codes)
    print(f"\n=== {name}  ({path}) ===")
    print(f"  {K} concurrent requests, {MP}MP image each | {ok}/{K} ok")
    print(f"  total wall time : {wall:8.0f} ms   <- how long all {K} took together")
    print(f"  request latency : p50 {pct(req_lat,50):6.0f} ms   p99 {pct(req_lat,99):6.0f} ms")
    print(f"  /healthz DURING : p50 {pct(healthz_lat,50):6.1f} ms   "
          f"MAX {max([x for x in healthz_lat if x==x] or [0]):6.0f} ms   "
          f"<- canary; high = loop frozen")


async def main():
    if not await wait_ready():
        print("server not reachable on", BASE); sys.exit(1)
    body = make_image(MP)
    print(f"test image: {MP}MP, {len(body)/1024:.0f} KB on disk")
    await run_endpoint("BLOCKING (bug: decode on event loop)", "/predict_blocking", body)
    await asyncio.sleep(0.3)
    await run_endpoint("OFFLOADED (fix: decode in thread pool)", "/predict_offloaded", body)
    print("\ninterpretation:")
    print("  blocking  -> total ~= K x decode (serialized) AND /healthz frozen the whole time")
    print("  offloaded -> total collapses (real parallelism) AND /healthz stays ~instant")


if __name__ == "__main__":
    asyncio.run(main())
