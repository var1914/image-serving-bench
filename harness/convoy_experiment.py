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


async def calibrate_service_ms(target, body, n=20):
    """Measure the REAL per-request decode+resize time on THIS server (read from its
    JSON response), so capacity/rho reflect the pod, not the dev laptop. Median of n."""
    times = []
    async with httpx.AsyncClient() as c:
        for _ in range(3):                                   # warm a few first
            try:
                await c.post(target, content=body, timeout=60)
            except Exception:
                pass
        for _ in range(n):
            try:
                j = (await c.post(target, content=body, timeout=60)).json()
                times.append(float(j.get("decode_ms", 0)) + float(j.get("resize_ms", 0)))
            except Exception:
                pass
    return sorted(times)[len(times) // 2] if times else 0.0


async def measure(a, cache, rng, rho, name, dist):
    """Run one condition end-to-end; print and return its result dict."""
    mps = [mp for mp, _ in dist]; ws = [w for _, w in dist]
    imgs = [cache[mp] for mp in mps]
    mean_mp = sum(w * m for w, m in zip(ws, mps))
    print(f"\n>>> {name}: mean {mean_mp:.2f} MP, Cs^2={cs2(mps, ws):.2f} | {a.rps:g} rps x {a.duration:g}s")
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=a.max_conns + 8)) as client:
        rows = await run_condition(client, a.target, imgs, mps, ws, a.rps, a.duration, a.max_conns, rng)
    lat = [r[0] * 1e3 for r in rows if r[2] == 200]
    om = [r[1] * 1e3 for r in rows]
    good = sum(1 for x in lat if x <= a.slo_ms) / max(1, len(rows)) * 100
    res = {"name": name, "n": len(rows), "ok": len(lat), "mean_mp": mean_mp,
           "cs2": cs2(mps, ws), "mp_per_s": mean_mp * a.rps, "rho": rho,
           "achieved_rps": len(rows) / a.duration,
           "p50": pct(lat, 50), "p99": pct(lat, 99), "p999": pct(lat, 99.9),
           "goodput": good, "omission_p99": pct(om, 99)}
    print(f"    p99={res['p99']:.0f}ms  goodput={good:.1f}%  ok={len(lat)}/{len(rows)}  "
          f"achieved={res['achieved_rps']:.0f}/{a.rps:g} rps  om_p99={pct(om, 99):.0f}ms")
    return res


async def main_async(a):
    root = a.target.rsplit("/", 1)[0]
    if not await wait_ready(root):
        print("server not reachable at", a.target); return
    rng = __import__("random").Random(11)
    # pre-generate every unique MP used
    uniq = sorted({mp for _, dist in CONDITIONS for mp, _ in dist})
    cache = {mp: gen_image(mp) for mp in uniq}
    os.makedirs(a.out, exist_ok=True)

    # --- calibrate on THIS hardware, then guard against overload ---------------
    # Laptop-measured constants badly mis-estimate pod capacity: they once printed
    # "rho 0.73 [OK]" for an rps that actually overloaded C0 to p99=24s and OOM'd the
    # pod. So measure the real per-3MP service time from the server's own response and
    # ABORT if the requested rps would exceed capacity (unbounded decode backlog -> OOM).
    svc_ms = await calibrate_service_ms(a.target, cache[3.0])
    if svc_ms <= 0:
        svc_ms = service_ms(3.0)                       # fallback to model if calibration failed
    capacity = a.server_cores * 1000.0 / svc_ms        # req/s the decode path can sustain
    rho = a.rps / capacity
    rec = 0.8 * capacity
    print(f"calibrated: 3 MP service ~= {svc_ms:.1f} ms/req -> capacity ~= {capacity:.0f} rps "
          f"on {a.server_cores} cores")
    print(f"operating point: {a.rps:g} rps -> rho ~= {rho:.2f}  (want 0.7-0.9; ~{rec:.0f} rps hits 0.8)")
    if rho > 0.95 and not a.force:
        print(f"\nABORT: --rps {a.rps:g} is ~{rho:.1f}x capacity -> this overloads and can OOM the pod "
              f"(unbounded decode backlog).\n       Rerun with --rps ~{rec:.0f} (rho 0.8), or --force to override.")
        return

    # --- warmup (discarded) — keeps the cold-start tax out of C0 ----------------
    # Warm at the SAME operating point as the conditions (mean 3 MP), NOT an
    # equal-weight mix over all sizes: {1,3,5,11,24} averages ~8.8 MP, which at the
    # target rps is ~2x capacity — the warmup would overload and take minutes to
    # drain instead of `warmup` seconds. Conditions run in sequence, so the big-image
    # decode path is warm well before C3 anyway.
    if a.warmup > 0:
        print(f"warmup: {a.warmup:g}s of discarded load at the operating mix (~3 MP)...")
        async with httpx.AsyncClient(limits=httpx.Limits(max_connections=a.max_conns + 8)) as client:
            await run_condition(client, a.target, [cache[3.0]], [3.0], [1.0],
                                a.rps, a.warmup, a.max_conns, rng)   # rows discarded

    results = []
    for name, dist in CONDITIONS:
        results.append(await measure(a, cache, rng, rho, name, dist))
        if a.settle:
            await asyncio.sleep(a.settle)

    # Drift sentinel — fixes the fixed-order confound. Conditions run C0..C3 with
    # variance rising by position, so any time-drift (memory growth, thermal, further
    # warming) would look like a variance effect. Re-run C0 last: if its p99 ~= the
    # first C0, there was no drift and the rising tax is really variance.
    if not a.no_recheck:
        results.append(await measure(a, cache, rng, rho, "C0_recheck", CONDITIONS[0][1]))

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
    fails = [r["name"] for r in results if r.get("ok", r["n"]) < r["n"]]
    if fails:
        print(f"  WARN: requests failed/timed out in {fails} -> p99 there EXCLUDES them "
              f"(tail understated). Lower --rps.")
    rc = next((r for r in results if r["name"] == "C0_recheck"), None)
    if rc and base:
        drift = rc["p99"] / base
        print(f"  drift check: C0_recheck p99 = {drift:.2f}x of C0 start -> "
              + ("negligible; the tax reflects variance, not run order."
                 if drift <= 1.2 else
                 "SIGNIFICANT; fixed order means some 'tax' may be time-drift. Rerun (shuffle order)."))
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
    ap.add_argument("--no-recheck", action="store_true",
                    help="skip the C0 re-run at the end (the time-drift sentinel)")
    ap.add_argument("--force", action="store_true",
                    help="run even if calibrated rho exceeds ~0.95x capacity (risks OOM)")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
