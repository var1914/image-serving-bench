#!/usr/bin/env python3
"""Open-loop load generator (coordinated-omission-safe).

THE ONE THING A SERVING BENCHMARK MUST GET RIGHT:
Requests are emitted on a SCHEDULE (Poisson or fixed rate), NOT "send the next one
after the previous returns." Closed-loop tools (ab, wrk without -R, locust default)
slow their own sending when the server slows -> the tail latency vanishes from the
numbers. That is *coordinated omission*. We emit at a target arrival process and
record (actual_send - scheduled_send); if that gap grows, the generator itself fell
behind and the tail is understated -> raise --rps headroom or --max-conns.

Also generates a heavy-tailed payload mix in-memory (like real uploads) so the
server's pw_payload_megapixels histogram shows a realistic spread in Grafana.

Usage:
  .venv/bin/python harness/loadgen.py --rps 40 --duration 60
  .venv/bin/python harness/loadgen.py --rps 40 --duration 60 --path /predict_blocking
"""
from __future__ import annotations
import argparse, asyncio, csv, io, os, random, time
import httpx
from PIL import Image
import numpy as np


def make_mix(seed: int = 7):
    """~12 images spanning a heavy-tailed megapixel range; weighted toward small,
    like real user uploads (median ~1 MP, long tail to 24 MP)."""
    rng = random.Random(seed)
    specs = [0.3, 0.5, 0.8, 1.0, 1.2, 2.0, 3.0, 5.0, 8.0, 12.0, 18.0, 24.0]
    weights = [10, 12, 12, 14, 14, 10, 8, 6, 5, 4, 3, 2]     # small dominates
    imgs = []
    for mp in specs:
        side = int((mp * 1e6) ** 0.5)
        xr = np.linspace(0, 255, side, dtype="uint8")
        arr = np.stack([np.broadcast_to(xr, (side, side)),
                        np.broadcast_to(xr[:, None], (side, side)),
                        np.broadcast_to(xr, (side, side)) // 2], axis=-1)
        buf = io.BytesIO(); Image.fromarray(arr, "RGB").save(buf, "JPEG", quality=85)
        imgs.append(buf.getvalue())
    return imgs, weights, specs, rng


def arrivals(pattern, rps, duration, rng):
    """Yield relative send times. poisson=exponential gaps; fixed=constant."""
    t = 0.0
    while t < duration:
        yield t
        t += rng.expovariate(rps) if pattern == "poisson" else 1.0 / rps

def target_mp(t, duration, lo, hi):
    """Linear ramp of the *mean* payload size. t is relative seconds."""
    return lo + (hi - lo) * min(1.0, t / duration)


def pick_payload(imgs, specs, weights, tgt, mode, rng):
    """Return (body, mp_of_body) for a request whose target mean is `tgt` MP."""
    if tgt is None:                                    # no sweep -> original static mix
        i = rng.choices(range(len(imgs)), weights=weights)[0]
        return imgs[i], specs[i]
    if mode == "nearest":
        i = min(range(len(specs)), key=lambda j: abs(specs[j] - tgt))
        return imgs[i], specs[i]
    near = sorted(range(len(specs)), key=lambda j: abs(specs[j] - tgt))[:3]
    w = [1.0 / (abs(specs[j] - tgt) + 0.25) for j in near]
    i = rng.choices(near, weights=w)[0]
    return imgs[i], specs[i]

async def wait_ready(base, timeout=20):
    async with httpx.AsyncClient() as c:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            try:
                if (await c.get(base + "/healthz", timeout=1)).status_code == 200:
                    return True
            except Exception:
                pass
            await asyncio.sleep(0.15)
    return False


def pct(a, p):
    a = sorted(a); return a[min(len(a) - 1, int(p / 100 * len(a)))] if a else float("nan")


async def run(a):
    base = a.target.rstrip("/") + ""
    root = a.target.rsplit("/", 1)[0]
    if not await wait_ready(root):
        print("server not reachable; start SUT-A first"); return
    imgs, weights, specs, rng = make_mix()
    rows, sem = [], asyncio.Semaphore(a.max_conns)

    async def fire(client, body, sched, t0, tgt, mp):
        async with sem:
            actual = time.perf_counter()
            omission = actual - (t0 + sched)          # >0 => we fell behind (backpressure)
            s = time.perf_counter()
            try:
                r = await client.post(a.target, content=body, timeout=a.timeout)
                code, err = r.status_code, ""
            except Exception as e:  # a slow payload timing out is data, not a crash
                code, err = 0, type(e).__name__
            rows.append({"t_rel": round(sched, 4),
                         "target_mp": round(tgt, 2) if tgt is not None else "",
                         "sent_mp": mp,
                         "lat_s": round(time.perf_counter() - s, 6),
                         "omission_s": round(omission, 6), "code": code, "err": err})

    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=a.max_conns + 8)) as client:
        t0 = time.perf_counter(); tasks = []
        for rel in arrivals(a.pattern, a.rps, a.duration, rng):
            now = time.perf_counter()
            if t0 + rel > now:
                await asyncio.sleep(t0 + rel - now)
            tgt = target_mp(rel, a.duration, *a.mix_sweep) if a.mix_sweep else None
            body, mp = pick_payload(imgs, specs, weights, tgt, a.mix_mode, rng)
            tasks.append(asyncio.create_task(fire(client, body, rel, t0, tgt, mp)))
        await asyncio.gather(*tasks)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "requests.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    lat = [r["lat_s"] * 1e3 for r in rows if r["code"] == 200]
    om  = [r["omission_s"] * 1e3 for r in rows]
    slo = a.slo_ms
    good = sum(1 for x in lat if x <= slo) / max(1, len(rows)) * 100
    print(f"\n[loadgen] {len(rows)} reqs @ {a.rps} rps for {a.duration}s -> {a.target}")
    print(f"  latency   p50 {pct(lat,50):6.0f}ms  p99 {pct(lat,99):6.0f}ms  p999 {pct(lat,99.9):6.0f}ms")
    print(f"  goodput@{slo:.0f}ms : {good:5.1f}%   ok={len(lat)}/{len(rows)}")
    print(f"  omission  p99 {pct(om,99):6.0f}ms  MAX {max(om or [0]):6.0f}ms  "
          f"(near 0 = generator kept up; big = raise --max-conns / lower --rps)")

    if a.mix_sweep:
        rows_t = sorted(rows, key=lambda r: r["t_rel"])
        nb, per = 10, max(1, len(rows) // 10)
        print("\n  target_MP     n     p50     p99  goodput  omission_p99")
        for b in range(nb):
            chunk = rows_t[b*per : (b+1)*per if b < nb-1 else None]
            if not chunk: continue
            l = [r["lat_s"]*1e3 for r in chunk if r["code"] == 200]
            o = [r["omission_s"]*1e3 for r in chunk]
            g = sum(1 for x in l if x <= slo) / len(chunk) * 100
            print(f"  {chunk[len(chunk)//2]['target_mp']:>9} {len(chunk):>5} "
                    f"{pct(l,50):>7.0f} {pct(l,99):>7.0f} {g:>7.1f}% {pct(o,99):>13.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="http://127.0.0.1:8099/predict_offloaded")
    ap.add_argument("--pattern", choices=["poisson", "fixed"], default="poisson")
    ap.add_argument("--rps", type=float, default=40)
    ap.add_argument("--duration", type=float, default=60)
    ap.add_argument("--max-conns", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--slo-ms", type=float, default=150)
    ap.add_argument("--out", default="results/loadgen")
    ap.add_argument("--mix-sweep", default=None,
                    type=lambda s: tuple(float(x) for x in s.split(":")),
                    help="ramp mean megapixels over --duration, e.g. 1:24")
    ap.add_argument("--mix-mode", choices=["nearest", "window"], default="nearest")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
