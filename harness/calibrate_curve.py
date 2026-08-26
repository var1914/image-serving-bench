#!/usr/bin/env python3
"""Calibrate the header->cost model on the TARGET hardware.

This is both the operating-point fix and the core of the header-cost contribution:
it measures real per-request decode+resize time across a grid of (megapixels x format),
reading the server's OWN reported timings, then fits  service_ms ~= a*MP + b  per format.
That a,b is exactly the model a caller could evaluate from header bytes (which give MP and
format) WITHOUT decoding — and it yields the real MP/s capacity to size any later run.

Low, sequential load only — no overload, safe to run first. Run the server first, then:
  python3 harness/calibrate_curve.py --server-cores <effective cores>   # see server startup log
"""
import argparse, asyncio, io, json, math, os, statistics
import httpx
from PIL import Image
import numpy as np

MPS = [0.3, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]
FORMATS = ["jpeg", "png", "webp"]


def effective_cores() -> int:
    """Container-aware core count (cgroup quota), same logic as the server."""
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
        if quota != "max":
            return max(1, math.floor(int(quota) / int(period)))
    except Exception:
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            q = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            p = int(f.read())
        if q > 0:
            return max(1, math.floor(q / p))
    except Exception:
        pass
    return os.cpu_count() or 1


def gen(mp: float, fmt: str) -> bytes:
    side = max(16, int((mp * 1e6) ** 0.5))
    xr = np.linspace(0, 255, side, dtype="uint8")
    arr = np.stack([np.broadcast_to(xr, (side, side)),
                    np.broadcast_to(xr[:, None], (side, side)),
                    np.broadcast_to(xr, (side, side)) // 2], axis=-1)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, "JPEG" if fmt == "jpeg" else fmt.upper(), quality=85)
    return buf.getvalue()


async def sample(client, target, body, n):
    ts = []
    for _ in range(n):
        try:
            j = (await client.post(target, content=body, timeout=120)).json()
            ts.append(float(j.get("decode_ms", 0)) + float(j.get("resize_ms", 0)))
        except Exception:
            pass
    return statistics.median(ts) if ts else float("nan")


async def main_async(a):
    cores = a.server_cores or effective_cores()
    print(f"calibrating against {a.target} | assuming {cores} effective server cores\n")
    rows = []
    async with httpx.AsyncClient() as client:
        try:
            for _ in range(3):                              # warm
                await client.post(a.target, content=gen(1.0, "jpeg"), timeout=60)
        except Exception:
            print("server not reachable at", a.target); return
        print(f"  {'fmt':4} {'MP':>5} {'on-disk':>9} {'service_ms':>11}")
        for fmt in FORMATS:
            for mp in MPS:
                body = gen(mp, fmt)
                ms = await sample(client, a.target, body, a.samples)
                rows.append({"format": fmt, "mp": mp, "bytes": len(body), "service_ms": round(ms, 2)})
                print(f"  {fmt:4} {mp:5.1f} {len(body)/1024:8.0f}K {ms:11.2f}")

    print(f"\nfit  service_ms ~= a*MP + b  (per format), capacity on {cores} cores:")
    fits = {}
    for fmt in FORMATS:
        xs = np.array([r["mp"] for r in rows if r["format"] == fmt])
        ys = np.array([r["service_ms"] for r in rows if r["format"] == fmt])
        A = np.vstack([xs, np.ones_like(xs)]).T
        (a_, b_), *_ = np.linalg.lstsq(A, ys, rcond=None)
        pred = A @ [a_, b_]
        ss, tot = ((ys - pred) ** 2).sum(), ((ys - ys.mean()) ** 2).sum()
        r2 = 1 - ss / tot if tot else float("nan")
        mpps = cores * 1000.0 / a_ if a_ > 0 else float("nan")     # MP/s at full CPU
        fits[fmt] = {"a_ms_per_mp": round(a_, 3), "b_ms": round(b_, 2),
                     "r2": round(r2, 4), "capacity_mp_per_s": round(mpps, 0)}
        print(f"  {fmt:4}: {a_:6.2f} ms/MP + {b_:6.2f}   R2={r2:.3f}   capacity ~= {mpps:6.0f} MP/s")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "calibration.json"), "w") as f:
        json.dump({"cores": cores, "rows": rows, "fits": fits}, f, indent=2)
    jpeg = fits.get("jpeg", {})
    print(f"\nsaved -> {a.out}/calibration.json")
    print("This a*MP+b IS the header->cost model (read MP+format from the header, predict cost).")
    if jpeg.get("capacity_mp_per_s"):
        cap = jpeg["capacity_mp_per_s"]
        print(f"To set a later run to rho~0.8 at mean 3 MP: rps ~= {0.8*cap/3:.0f} "
              f"(0.8 * {cap:.0f} MP/s / 3 MP).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="http://127.0.0.1:8099/predict_offloaded")
    ap.add_argument("--samples", type=int, default=15)
    ap.add_argument("--server-cores", type=int, default=0,
                    help="effective server cores; 0 = auto-detect from cgroup on this box")
    ap.add_argument("--out", default="results/calibration")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
