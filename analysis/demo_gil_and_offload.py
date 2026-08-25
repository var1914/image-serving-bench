#!/usr/bin/env python3
"""The GIL, made visible: one request traced phase by phase.

Two experiments:
  A) TRACE one request's journey: it arrives on the event-loop thread, gets
     offloaded to a worker thread (same process, same GIL), runs the CPU work,
     and returns to the loop. We stamp thread name + thread id + pid at each phase
     so you literally watch it move.
  B) PROVE when the GIL bites: run the same batch of work three ways — serial,
     threads, processes — for TWO kinds of work:
       - GIL-HOLDING  : a pure-Python CPU loop (only Python bytecode)
       - GIL-RELEASING: a Pillow decode (C code that drops the GIL while it runs)
     The speedup table is the GIL's fingerprint.

Run:  .venv/bin/python analysis/demo_gil_and_offload.py
"""
from __future__ import annotations
import asyncio, io, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from PIL import Image

T0 = time.perf_counter()

def stamp(phase: str) -> None:
    t = (time.perf_counter() - T0) * 1e3
    th = threading.current_thread()
    print(f"    [{t:8.1f} ms] {phase:38} thread={th.name:20} pid={os.getpid()}")

# ---- work functions. MUST be module-level so ProcessPoolExecutor (spawn) can
#      re-import and pickle them. ----------------------------------------------

def py_cpu(n: int) -> int:
    """GIL-HOLDING work: pure-Python loop. The interpreter holds the GIL the
    entire time — no other Python thread can run bytecode meanwhile."""
    total = 0
    for i in range(n):
        total += (i * i) & 15
    return total

def decode(jpeg: bytes):
    """GIL-RELEASING work: Pillow decodes in C and RELEASES the GIL while doing
    it, so other Python threads can run concurrently. (Same as yesterday's fix.)"""
    img = Image.open(io.BytesIO(jpeg)).convert("RGB"); img.load()
    return img.size

def decode_traced(jpeg: bytes):
    """Same decode, but it announces which thread/pid it's running on."""
    stamp(">>> INSIDE worker: about to decode")
    img = Image.open(io.BytesIO(jpeg)).convert("RGB"); img.load()
    stamp(">>> INSIDE worker: C decode returned (GIL was released during it)")
    return img.size

def where_am_i(_):
    """Runs in a process-pool worker; reports its pid back to the parent."""
    return os.getpid()

def make_jpeg(mp: float) -> bytes:
    side = int((mp * 1e6) ** 0.5)
    xr = np.linspace(0, 255, side, dtype="uint8")
    arr = np.stack([np.broadcast_to(xr, (side, side)),
                    np.broadcast_to(xr[:, None], (side, side)),
                    np.broadcast_to(xr, (side, side)) // 2], axis=-1)
    buf = io.BytesIO(); Image.fromarray(arr, "RGB").save(buf, "JPEG", quality=85)
    return buf.getvalue()


# ============================ EXPERIMENT A ==================================
async def trace_journey(jpeg: bytes):
    print("\n" + "=" * 78)
    print("A) ONE REQUEST, TRACED — watch it leave the event loop and come back")
    print("=" * 78)
    tpool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="decode-worker")
    stamp("request arrives on the EVENT LOOP")
    loop = asyncio.get_running_loop()
    stamp("offloading decode via run_in_executor  --->")
    size = await loop.run_in_executor(tpool, decode_traced, jpeg)   # exactly what /predict_offloaded does
    stamp("back on the EVENT LOOP, size=%s" % (size,))
    tpool.shutdown(wait=True)
    print("\n  ^ note: the worker line has a DIFFERENT thread name but the SAME pid")
    print("    -> a thread is a lane inside the same process; it shares the one GIL.")

    # contrast: the same work in a separate PROCESS has its own pid + its own GIL
    with ProcessPoolExecutor(max_workers=1) as ppool:
        worker_pid = ppool.submit(where_am_i, None).result()
    print(f"\n  same work in a PROCESS pool ran in pid={worker_pid} "
          f"(parent pid={os.getpid()}) -> different process, its OWN GIL.")


# ============================ EXPERIMENT B ==================================
def timed(fn):
    t = time.perf_counter(); fn(); return (time.perf_counter() - t) * 1e3

def batch_serial(fn, arg, k):   return lambda: [fn(arg) for _ in range(k)]
def batch_pool(pool, fn, arg, k): return lambda: list(pool.map(fn, [arg] * k))

def experiment_b(jpeg: bytes):
    print("\n" + "=" * 78)
    print("B) WHEN DOES THE GIL BITE?  same batch, 3 ways, 2 kinds of work")
    print("=" * 78)
    cores = os.cpu_count() or 4
    K = min(4, cores)
    print(f"  machine: {cores} CPU cores | batch size K={K} | "
          f"GIL switch interval = {sys.getswitchinterval()*1000:.0f} ms")

    # calibrate the pure-Python loop to ~50 ms per call so both works are comparable
    probe_n = 1_000_000
    one = timed(lambda: py_cpu(probe_n))
    PY_N = max(1, int(probe_n * 50.0 / one))
    print(f"  pure-Python loop sized to ~50 ms/call (PY_N={PY_N:,})")

    tpool = ThreadPoolExecutor(max_workers=K)
    ppool = ProcessPoolExecutor(max_workers=K)
    # warm both pools so spawn / thread-start cost is excluded from the numbers
    list(tpool.map(py_cpu, [1] * K)); list(ppool.map(py_cpu, [1] * K))

    rows = []
    for label, fn, arg in [("GIL-HOLDING  (pure-Python loop)", py_cpu, PY_N),
                           ("GIL-RELEASING (Pillow decode)  ", decode, jpeg)]:
        s = timed(batch_serial(fn, arg, K))
        t = timed(batch_pool(tpool, fn, arg, K))
        p = timed(batch_pool(ppool, fn, arg, K))
        rows.append((label, s, t, p))
    tpool.shutdown(); ppool.shutdown()

    print(f"\n  {'work type':34} {'serial':>9} {'threads':>9} {'processes':>10}")
    print("  " + "-" * 64)
    for label, s, t, p in rows:
        print(f"  {label:34} {s:7.0f}ms {t:7.0f}ms {p:8.0f}ms")
    print(f"  {'':34} {'':>9} {'speedup vs serial ->':>30}")
    for label, s, t, p in rows:
        print(f"  {label:34} {'1.0x':>9} {s/t:7.1f}x {s/p:8.1f}x")

    print("\n  read it:")
    print("   - pure-Python across THREADS ~= serial (no speedup): the GIL lets")
    print("     only one thread run bytecode at a time. Threads don't help CPU-bound")
    print("     Python. PROCESSES do (each has its own interpreter + GIL).")
    print("   - Pillow decode across THREADS already speeds up: its C code releases")
    print("     the GIL, so threads overlap. That's why yesterday's threadpool fix worked.")
    print("   - takeaway: threadpool for GIL-releasing work (decode, numpy, I/O);")
    print("     worker PROCESSES (uvicorn --workers / gunicorn) for Python CPU work.")


async def main():
    jpeg = make_jpeg(6)
    await trace_journey(jpeg)
    experiment_b(jpeg)


if __name__ == "__main__":
    asyncio.run(main())
