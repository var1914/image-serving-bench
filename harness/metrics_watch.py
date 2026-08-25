#!/usr/bin/env python3
"""Terminal metrics watcher — a no-Docker, no-Grafana live view of the server.

Polls the server's /metrics endpoint every --interval seconds and prints:
  rate (req/s), in-flight, windowed p50/p99 latency, and mean payload megapixels.
"Windowed" means computed over the delta since the last poll (like Prometheus
rate()), so it reflects what is happening right now. p50/p99 are computed from the
histogram buckets by hand — the same interpolation histogram_quantile() does —
never by averaging percentiles.

Stdlib only (no pip installs, no Docker). Useful for tuning --rps toward rho~0.8:
watch in-flight and p99 climb as you raise the load.

Usage: python3 harness/metrics_watch.py --url http://localhost:8099/metrics
"""
import argparse, time, urllib.request
from collections import defaultdict


def _le(name):
    """Pull the le="..." bound out of a histogram bucket line's labels."""
    i = name.find('le="')
    if i < 0:
        return None
    s = name[i + 4:name.find('"', i + 4)]
    return float("inf") if s in ("+Inf", "Inf") else float(s)


def parse(text):
    reqs = inflight = dur_c = dur_s = mp_c = mp_s = 0.0
    dur_b, mp_b = defaultdict(float), defaultdict(float)
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        name, _, val = line.rpartition(" ")
        try:
            v = float(val)
        except ValueError:
            continue
        if name.startswith("pw_inflight"):
            inflight = v
        elif name.startswith("pw_requests_total") and "predict" in name:
            reqs += v
        elif name.startswith("pw_request_duration_seconds_bucket") and "predict" in name:
            le = _le(name)
            if le is not None:
                dur_b[le] += v
        elif name.startswith("pw_request_duration_seconds_count") and "predict" in name:
            dur_c += v
        elif name.startswith("pw_request_duration_seconds_sum") and "predict" in name:
            dur_s += v
        elif name.startswith("pw_payload_megapixels_bucket"):
            le = _le(name)
            if le is not None:
                mp_b[le] += v
        elif name.startswith("pw_payload_megapixels_count"):
            mp_c += v
        elif name.startswith("pw_payload_megapixels_sum"):
            mp_s += v
    return dict(reqs=reqs, inflight=inflight, dur_b=dur_b, dur_c=dur_c, dur_s=dur_s,
                mp_c=mp_c, mp_s=mp_s)


def quantile(bucket_delta, q):
    """histogram_quantile over cumulative Prometheus buckets (as deltas)."""
    if not bucket_delta:
        return float("nan")
    les = sorted(bucket_delta)
    total = bucket_delta[les[-1]]                 # the +Inf bucket holds the total
    if total <= 0:
        return float("nan")
    target = q * total
    prev_le = prev_c = 0.0
    for le in les:
        c = bucket_delta[le]
        if c >= target:
            if le == float("inf"):
                return prev_le
            if c == prev_c:
                return le
            return prev_le + (target - prev_c) / (c - prev_c) * (le - prev_le)
        prev_le, prev_c = le, c
    return les[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8099/metrics")
    ap.add_argument("--interval", type=float, default=2.0)
    a = ap.parse_args()
    prev = prev_t = None
    print(f"{'time':>8} {'req/s':>7} {'inflight':>9} {'p50 ms':>7} {'p99 ms':>7} {'mean MP':>8}", flush=True)
    try:
        while True:
            try:
                now = time.perf_counter()
                cur = parse(urllib.request.urlopen(a.url, timeout=5).read().decode())
            except Exception:
                print("  (waiting for server /metrics ...)", flush=True)
                time.sleep(a.interval)
                continue
            if prev is not None:
                dt = now - prev_t or 1e-9
                rate = (cur["reqs"] - prev["reqs"]) / dt
                db = {le: cur["dur_b"][le] - prev["dur_b"].get(le, 0.0) for le in cur["dur_b"]}
                p50 = quantile(db, 0.50) * 1e3
                p99 = quantile(db, 0.99) * 1e3
                dmpc = cur["mp_c"] - prev["mp_c"]
                mean_mp = (cur["mp_s"] - prev["mp_s"]) / dmpc if dmpc > 0 else float("nan")
                print(f"{time.strftime('%H:%M:%S'):>8} {rate:7.0f} {cur['inflight']:9.0f} "
                      f"{p50:7.0f} {p99:7.0f} {mean_mp:8.1f}", flush=True)
            prev, prev_t = cur, now
            time.sleep(a.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
