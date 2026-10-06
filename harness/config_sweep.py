#!/usr/bin/env python3
"""Stress SUT-C (ONNX ResNet-50) under different CPU-sharing configs. Pod-ready.

For each config: start the server pinned to --server-cpus, warm it, record its OS
thread count, then climb an open-loop load ladder (loadgen pinned to --client-cpus,
fixed 1 MP photo-like images) until it breaks. Per step it also records how much the
kernel throttled the container (cgroup cpu.stat), which is how a CPU quota bites.

Configs (N = cores the server may use: the --server-cpus count, else the container quota):
  a_default  1 request at a time, ORT default threads (one per HOST core -> the trap)
  a_sized    1 request at a time, N threads
  b          N requests at once, 1 thread each
  c_default  N requests at once, each with default threads (trap x N)
  c_nospin   as c_default, but ORT threads sleep instead of spin-waiting

Usage on the pod (8.5-core quota: server on 0-6, client on 7):
  python harness/config_sweep.py --server-cpus 0-6 --client-cpus 7
Locally (no taskset, co-located client -> treat as a smoke test only):
  python harness/config_sweep.py --configs b --ladder 5,10 --duration 5
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

from cpuinfo import describe, effective_cores, thread_count, throttle_stats

PORT = 8100
PY = sys.executable


def cpus_in(spec):
    """'0-6,8' -> 8"""
    n = 0
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        n += int(hi) - int(lo) + 1 if hi else 1
    return n


def configs(n):
    # name: (workers, ort_threads, spin, description)
    return {
        "a_default": (1, 0, "1", "1 request at a time, ORT default threads (host-sized)"),
        "a_sized": (1, n, "1", f"1 request at a time, {n} threads"),
        "b": (n, 1, "1", f"{n} requests at once, 1 thread each"),
        "c_default": (n, 0, "1", f"{n} requests at once, each with default threads"),
        "c_nospin": (n, 0, "0", "as c_default, threads sleep instead of spin-wait"),
    }


def pinned(cmd, cpus):
    if cpus and shutil.which("taskset"):
        return ["taskset", "-c", cpus] + cmd
    return cmd


def pct(a, p):
    a = sorted(a)
    return a[min(len(a) - 1, int(p / 100 * len(a)))] if a else float("nan")


def wait_healthy(timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/healthz", timeout=1)
            return True
        except OSError:
            time.sleep(0.5)
    return False


def run_load(rps, dur, a, out):
    cmd = [PY, "harness/loadgen.py", "--target", f"http://127.0.0.1:{PORT}/predict",
           "--rps", str(rps), "--duration", str(dur), "--content", "photo",
           "--fixed-mp", str(a.mp), "--slo-ms", str(a.slo_ms), "--max-conns", "256", "--out", out]
    subprocess.run(pinned(cmd, a.client_cpus), stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, check=False)


def step(cfg, rps, a, pid):
    out = f"{a.out}/{cfg}/rps{rps}"
    before = throttle_stats()
    run_load(rps, a.duration, a, out)
    after = throttle_stats()
    os_threads = thread_count(pid)        # request-pool threads are created lazily, so read under load
    with open(os.path.join(out, "requests.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    lat = [float(r["lat_s"]) * 1e3 for r in rows if r["code"] == "200"]
    om = [float(r["omission_s"]) * 1e3 for r in rows]
    good = sum(1 for x in lat if x <= a.slo_ms) / max(1, len(rows)) * 100
    thr = (after[0] - before[0], after[1] - before[1]) if before and after else (None, None)
    return {"config": cfg, "rps": rps, "n": len(rows), "p50_ms": pct(lat, 50),
            "p99_ms": pct(lat, 99), "goodput_pct": good, "omission_p99_ms": pct(om, 99),
            "os_threads": os_threads, "throttled_periods": thr[0], "throttled_ms": thr[1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="a_default,a_sized,b,c_default,c_nospin")
    ap.add_argument("--ladder", default="10,20,40,60,80,100,130,160,200,250,300,400")
    ap.add_argument("--duration", type=float, default=20)
    ap.add_argument("--warmup", type=float, default=5)
    ap.add_argument("--mp", type=float, default=1.0)
    ap.add_argument("--slo-ms", type=float, default=150)
    ap.add_argument("--server-cpus", default=None, help="taskset list for the server, e.g. 0-6")
    ap.add_argument("--client-cpus", default=None, help="taskset list for the load generator, e.g. 7")
    ap.add_argument("--out", default="results/sweep")
    a = ap.parse_args()

    n = cpus_in(a.server_cpus) if a.server_cpus else effective_cores()
    ladder = [int(x) for x in a.ladder.split(",")]
    table = configs(n)
    print(f"CPU: {describe()}")
    print(f"server cores N={n} (cpus {a.server_cpus or 'unpinned'}), "
          f"client cpus {a.client_cpus or 'unpinned'}")
    if not shutil.which("taskset"):
        print("note: no taskset here -> server and client share cores; treat as a smoke test")

    steps, summary = [], []
    for cfg in a.configs.split(","):
        workers, threads, spin, desc = table[cfg]
        env = dict(os.environ, PW_WORKERS=str(workers), PW_ORT_THREADS=str(threads),
                   PW_ORT_SPIN=spin)
        cmd = [PY, "-m", "uvicorn", "sut.onnx_server.server:app", "--host", "127.0.0.1",
               "--port", str(PORT), "--log-level", "warning"]
        srv = subprocess.Popen(pinned(cmd, a.server_cpus), env=env,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            if not wait_healthy():
                print(f"[{cfg}] server failed to start")
                continue
            run_load(ladder[0], a.warmup, a, f"{a.out}/{cfg}/warmup")      # discarded
            print(f"\n=== {cfg}: {desc} | OS threads after warmup: {thread_count(srv.pid)} ===")
            print(f"{'rps':>5} {'p50 ms':>7} {'p99 ms':>8} {'good%':>6} {'omis p99':>9} "
                  f"{'OS thr':>7} {'throttled':>10}")
            best, best_p99, peak_threads = 0, None, 0
            for rps in ladder:
                r = step(cfg, rps, a, srv.pid)
                steps.append(r)
                peak_threads = max(peak_threads, r["os_threads"] or 0)
                ok = r["goodput_pct"] >= 95 and r["omission_p99_ms"] < 50
                thr = "n/a" if r["throttled_ms"] is None else f"{r['throttled_ms']:.0f}ms"
                print(f"{rps:5d} {r['p50_ms']:7.0f} {r['p99_ms']:8.0f} {r['goodput_pct']:6.1f} "
                      f"{r['omission_p99_ms']:8.0f}ms {str(r['os_threads']):>7} {thr:>10}"
                      f"{'' if ok else '   <- broke'}")
                if not ok:
                    break
                best, best_p99 = rps, r["p99_ms"]
            summary.append({"config": cfg, "workers": workers, "ort_threads": threads or "default",
                            "spin": spin, "os_threads_peak": peak_threads, "max_rps": best,
                            "p99_at_max_ms": best_p99})
        finally:
            srv.terminate()
            srv.wait(timeout=15)
            time.sleep(2)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "steps.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(steps[0].keys()))
        w.writeheader()
        w.writerows(steps)
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"cpu": describe(), "server_cores": n, "summary": summary}, f, indent=2)

    print("\n=== max load held cleanly (goodput >= 95%, generator kept up) ===")
    print(f"{'config':10} {'workers':>7} {'threads':>8} {'spin':>4} {'OS thr':>7} {'max rps':>8} {'p99 ms':>7}")
    for s in summary:
        p99 = "-" if s["p99_at_max_ms"] is None else f"{s['p99_at_max_ms']:.0f}"
        print(f"{s['config']:10} {s['workers']:>7} {str(s['ort_threads']):>8} {s['spin']:>4} "
              f"{str(s['os_threads_peak']):>7} {s['max_rps']:>8} {p99:>7}")
    print(f"\nsaved -> {a.out}/steps.csv, {a.out}/summary.json")


if __name__ == "__main__":
    main()
