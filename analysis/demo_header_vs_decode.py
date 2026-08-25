#!/usr/bin/env python3
"""Lesson 1 demo: the header tells you the cost feature for free.

Generates one biggish image per format, then for each races:
  (a) header probe  -- pure-python, reads a few hundred bytes, no pixels
  (b) full decode   -- PIL actually reconstructs every pixel
Prints dims agreement + wall-time ratio. Run: .venv/bin/python analysis/demo_header_vs_decode.py
"""
import io, time, statistics, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "corpus"))
from imgprobe import probe                      # our zero-dep parser
from PIL import Image
import numpy as np

def make(fmt, w, h):
    arr = np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype="uint8")
    buf = io.BytesIO()
    img = Image.fromarray(arr, "RGB")
    if fmt == "jpeg": img.save(buf, "JPEG", quality=85)
    elif fmt == "png": img.save(buf, "PNG")
    else: img.save(buf, "WEBP", quality=85)
    return buf.getvalue()

def timeit(fn, reps):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return statistics.median(ts)

print(f"{'format':7} {'on-disk':>9} {'true dims':>12} {'probe dims':>12} "
      f"{'probe µs':>9} {'decode µs':>10} {'speedup':>8}")
print("-" * 74)
for fmt, w, h in [("jpeg", 4000, 3000), ("png", 4000, 3000), ("webp", 4000, 3000)]:
    data = make(fmt, w, h)
    pr = probe(data)
    t_probe  = timeit(lambda: probe(data), 2000)
    t_decode = timeit(lambda: Image.open(io.BytesIO(data)).convert("RGB").load(), 20)
    print(f"{fmt:7} {len(data)/1024:8.1f}K {f'{w}x{h}':>12} "
          f"{f'{pr.width}x{pr.height}':>12} {t_probe*1e6:9.1f} {t_decode*1e6:10.1f} "
          f"{t_decode/t_probe:7.0f}x")

print("\nTakeaway: same 12 MP image, three formats. The probe reads the SAME")
print("dimensions the decoder would, ~1000x faster, by touching a few hundred")
print("bytes instead of reconstructing 12 million pixels. That is the RQ3 fix:")
print("the caller hands you the cost driver in the header, before you pay to decode.")
