#!/usr/bin/env python3
"""Convoy Tax experiment — does payload-size VARIANCE inflate the tail at constant
average load, and does a real GIL-threadpool image server obey queueing theory?

Design (the confound control): every condition sends the SAME requests/sec AND the
SAME mean megapixels (=> same aggregate MP/s => same average CPU load). Only the
VARIANCE of payload size changes. So any change in p99 is pure lumpiness, not load.

Theory baseline (Kingman): queue wait ~ (Ca^2 + Cs^2)/2, where Cs = coefficient of
variation of SERVICE time. Service time ~ 2.2*MP + 0.66 ms (measured empirically). With
Poisson arrivals Ca^2 = 1, so predicted relative tail-inflation vs the zero-variance
condition is (1 + Cs^2). We compare MEASURED p99 tax to that prediction. Divergence
is the finding.

Runs a discarded warmup, then all conditions back-to-back, and prints the comparison
table plus a utilization (rho) proxy and a coordinated-omission trustworthiness gate.
Usage: python harness/convoy_experiment.py --rps 700 --duration 90 --warmup 20 --server-cores 7
"""
from __future__ import annotations
import argparse, asyncio, io, json, os, statistics, time
import httpx
from PIL import Image
import numpy as np

# Conditions: all mean(MP)=3.0 -> identical MP/s at a fixed rps. Variance climbs C0->C3.
CONDITIONS = [
    ("C0_uniform",   [(3.0, 1.00)]),                 # zero variance
    ("C1_low",       [(1.0, 0.50), (5.0, 0.50)]),    # mean 3.0
    ("C2_mid",       [(1.0, 0.80), (11.0, 0.20)]),   # mean 3.0
    ("C3_lumpy",     [(1.0, 0.913), (24.0, 0.087)]), # mean 3.0, a few monsters
]
MS_PER_MP, MS_FIXED = 2.2, 0.66   # service-time model (measured empirically)


def gen_image(mp: float) -> bytes:
    side = max(16, int((mp * 1e6) ** 0.5))
    xr = np.linspace(0, 255, side, dtype="uint8")
    arr = np.stack([np.broadcast_to(xr, (side, side)),
                    np.broadcast_to(xr[:, None], (side, side)),
                    np.broadcast_to(xr, (side, side)) // 2], axis=-1)
    buf = io.BytesIO(); Image.fromarray(arr, "RGB").save(buf, "JPEG", quality=85)
    return buf.getvalue()


def service_ms(mp): return MS_PER_MP * mp + MS_FIXED
def cs2(mps, ws):
    st = [service_ms(m) for m in mps]
    mean = sum(w * s for w, s in zip(ws, st))
    var = sum(w * (s - mean) ** 2 for w, s in zip(ws, st))
    return var / (mean ** 2) if mean else 0.0
def pct(a, p):
    a = sorted(a); return a[min(len(a) - 1, int(p / 100 * len(a)))] if a else float("nan")


async def wait_ready(root, timeout=20):
    async with httpx.AsyncClient() as c:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            try:
                if (await c.get(root + "/healthz", timeout=1)).status_code == 200:
                    return True
            except Exception:
                await asyncio.sleep(0.15)
    return False


async def run_condition(client, target, imgs, mps, ws, rps, duration, max_conns, rng):
    sem = asyncio.Semaphore(max_conns)
    rows = []

    async def fire(body, sched, t0):
        async with sem:
            omission = time.perf_counter() - (t0 + sched)
            s = time.perf_counter()
            try:
                r = await client.post(target, content=body, timeout=60)
                code = r.status_code
            except Exception as e:
                code = 0
            rows.append((time.perf_counter() - s, omission, code))

    idx = list(range(len(mps)))
    t0 = time.perf_counter(); t = 0.0; tasks = []
    while t < duration:
        now = time.perf_counter()
        if t0 + t > now:
            await asyncio.sleep(t0 + t - now)
        i = rng.choices(idx, weights=ws)[0]
        tasks.append(asyncio.create_task(fire(imgs[i], t, t0)))
        t += rng.expovariate(rps)
    await asyncio.gather(*tasks)
    return rows


async def main_async(a):
    root = a.target.rsplit("/", 1)[0]
    if not await wait_ready(root):
        print("server not reachable at", a.target); return
    rng = __import__("random").Random(11)
    # pre-generate every unique MP used
    uniq = sorted({mp for _, dist in CONDITIONS for mp, _ in dist})
    cache = {mp: gen_image(mp) for mp in uniq}
    os.makedirs(a.out, exist_ok=True)

    # --- operating point (utilization proxy) -----------------------------------
    # Every condition has mean 3.0 MP, so offered load is the same for all of them.
    # rho ~= (work offered per second) / (server capacity). work/s = rps x mean
    # service time; capacity = server_cores x 1000 ms/s. Aim for rho in 0.7-0.9:
    # high enough that a queue forms (so variance can bite), low enough that C0
    # isn't already melting down.
    svc_ms = service_ms(3.0)
    rho = a.rps * svc_ms / (a.server_cores * 1000)
    band = "OK" if 0.6 <= rho <= 0.92 else ("TOO LOW - raise --rps" if rho < 0.6 else "TOO HIGH - lower --rps")
    print(f"operating point: {a.rps:g} rps x {svc_ms:.1f} ms/req / {a.server_cores} cores "
          f"~= rho {rho:.2f}  [{band}]  (want 0.7-0.9)")

    # --- warmup (discarded) — keeps the cold-start tax out of C0 ----------------
    if a.warmup > 0:
        print(f"warmup: {a.warmup:g}s of discarded load (thread pool / allocator / decode paths)...")
        warm_imgs = list(cache.values()); warm_mps = list(cache.keys())
        async with httpx.AsyncClient(limits=httpx.Limits(max_connections=a.max_conns + 8)) as client:
            await run_condition(client, a.target, warm_imgs, warm_mps, [1] * len(warm_imgs),
                                a.rps, a.warmup, a.max_conns, rng)   # rows discarded

    results = []
    for name, dist in CONDITIONS:
        mps = [mp for mp, _ in dist]; ws = [w for _, w in dist]
        imgs = [cache[mp] for mp in mps]
        mean_mp = sum(w * m for w, m in zip(ws, mps))
        print(f"\n>>> {name}: mean {mean_mp:.2f} MP, Cs^2={cs2(mps, ws):.2f} "
              f"| {a.rps} rps for {a.duration}s  (watch Grafana now)")
        async with httpx.AsyncClient(limits=httpx.Limits(max_connections=a.max_conns + 8)) as client:
            rows = await run_condition(client, a.target, imgs, mps, ws, a.rps, a.duration, a.max_conns, rng)
        lat = [r[0] * 1e3 for r in rows if r[2] == 200]
        om  = [r[1] * 1e3 for r in rows]
        good = sum(1 for x in lat if x <= a.slo_ms) / max(1, len(rows)) * 100
        achieved_rps = len(rows) / a.duration
        results.append({"name": name, "n": len(rows), "mean_mp": mean_mp,
                        "cs2": cs2(mps, ws), "mp_per_s": mean_mp * a.rps, "rho": rho,
                        "achieved_rps": achieved_rps,
                        "p50": pct(lat, 50), "p99": pct(lat, 99), "p999": pct(lat, 99.9),
                        "goodput": good, "omission_p99": pct(om, 99)})
        print(f"    p99={results[-1]['p99']:.0f}ms  goodput={good:.1f}%  "
              f"achieved={achieved_rps:.0f}/{a.rps:g} rps  omission_p99={pct(om,99):.0f}ms")
        if a.settle:
            await asyncio.sleep(a.settle)

    base = results[0]["p99"]
    kbase = 1 + results[0]["cs2"]
    print("\n" + "=" * 84)
    print("CONVOY TAX — equal mean load, rising payload variance")
    print("=" * 84)
    print(f"{'cond':11} {'meanMP':>6} {'MP/s':>6} {'Cs^2':>6} {'p99 ms':>7} "
          f"{'good%':>6} {'tax(meas)':>10} {'tax(Kingman)':>13} {'om_p99':>7}")
    for r in results:
        meas = r["p99"] / base if base else float("nan")
        king = (1 + r["cs2"]) / kbase
        print(f"{r['name']:11} {r['mean_mp']:6.2f} {r['mp_per_s']:6.0f} {r['cs2']:6.2f} "
              f"{r['p99']:7.0f} {r['goodput']:6.1f} {meas:9.2f}x {king:12.2f}x {r['omission_p99']:6.0f}")
    print("\n  meas > Kingman  => the real server is WORSE than theory (GIL / head-of-line).")
    print("  meas ~ Kingman  => it obeys queueing theory; variance is the whole story.")
    print("  meas < Kingman  => decode's GIL release overlaps work better than M/M/c assumes.")
    print("  (check om_p99 is small in every row, else the generator fell behind -> rerun.)")
    bad_om = [r["name"] for r in results if r["omission_p99"] > 50]
    print(f"\n  operating rho ~= {rho:.2f}  ({'in band' if 0.6 <= rho <= 0.92 else 'OUT OF BAND - retune --rps'}).")
    print("  OK: omission low across all conditions -> tail numbers are trustworthy."
          if not bad_om else
          f"  WARN: high omission in {bad_om} -> those rows are contaminated; retune and rerun.")
    with open(os.path.join(a.out, "convoy_results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n  saved -> {a.out}/convoy_results.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="http://127.0.0.1:8099/predict_offloaded")
    ap.add_argument("--rps", type=float, default=60)
    ap.add_argument("--duration", type=float, default=90)
    ap.add_argument("--settle", type=float, default=5)
    ap.add_argument("--max-conns", type=int, default=256)
    ap.add_argument("--slo-ms", type=float, default=150)
    ap.add_argument("--out", default="results/convoy")
    ap.add_argument("--warmup", type=float, default=20,
                    help="seconds of discarded warmup load before C0 (avoids cold-start bias)")
    ap.add_argument("--server-cores", type=int, default=7,
                    help="cores the server is pinned to (taskset) — used for the rho utilization proxy")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
