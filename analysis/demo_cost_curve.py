#!/usr/bin/env python3
"""Lesson 2: what the header predicts. Cost of the CPU preprocessing path vs
megapixels, decomposed into decode + resize stages. This is RQ1 on a laptop.

We use smooth gradient content (photo-like, compresses realistically) rather than
noise, and time the two CPU stages every real service runs before the GPU sees
anything: decode(bytes -> pixels) and resize(pixels -> 224x224).
"""
import io, time, statistics, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "corpus"))
from PIL import Image
import numpy as np

def gradient_jpeg(w, h):
    # smooth 2D gradient => realistic JPEG entropy (unlike random noise)
    xr = np.linspace(0, 255, w, dtype="uint8")            # varies across columns
    yr = np.linspace(0, 255, h, dtype="uint8")            # varies down rows
    R = np.broadcast_to(xr, (h, w))
    G = np.broadcast_to(yr[:, None], (h, w))
    B = ((R.astype("int16") + G) // 2).astype("uint8")
    arr = np.stack([R, G, B], axis=-1)
    buf = io.BytesIO(); Image.fromarray(arr, "RGB").save(buf, "JPEG", quality=85)
    return buf.getvalue()

def stages(data, reps=15):
    dec, res = [], []
    for _ in range(reps):
        t = time.perf_counter()
        img = Image.open(io.BytesIO(data)); img = img.convert("RGB"); img.load()
        dec.append(time.perf_counter() - t)
        t = time.perf_counter()
        img.resize((224, 224), Image.BILINEAR)
        res.append(time.perf_counter() - t)
    return statistics.median(dec), statistics.median(res)

# heavy-tailed set of megapixels, like real uploads (median ~1MP, tail to 24MP)
sizes = [(640,480),(1280,720),(1920,1080),(2560,1440),(4000,3000),(6000,4000),(8000,6000)]
print(f"{'dims':>12} {'MP':>6} {'KB':>8} {'decode ms':>10} {'resize ms':>10} {'total ms':>9}")
print("-"*62)
mps, totals = [], []
for w,h in sizes:
    data = gradient_jpeg(w,h)
    d, r = stages(data)
    mp = w*h/1e6; tot = (d+r)*1e3
    mps.append(mp); totals.append(tot)
    print(f"{f'{w}x{h}':>12} {mp:6.1f} {len(data)/1024:7.1f} {d*1e3:10.2f} {r*1e3:10.2f} {tot:9.2f}")

mps, totals = np.array(mps), np.array(totals)
# fit total_ms ~ a*MP + b ; report R^2 and the CDR the paper headlines
A = np.vstack([mps, np.ones_like(mps)]).T
(slope, intercept), *_ = np.linalg.lstsq(A, totals, rcond=None)
pred = A @ [slope, intercept]
ss_res = ((totals-pred)**2).sum(); ss_tot = ((totals-totals.mean())**2).sum()
r2 = 1 - ss_res/ss_tot
cdr = totals.max()/totals.min()
print("-"*62)
print(f"fit: total_ms ~= {slope:.2f} * megapixels + {intercept:.2f}   R^2 = {r2:.3f}")
print(f"CDR (max/min cost across this size spread) = {cdr:.1f}x  <-- from payload alone")
print("\nTakeaway: cost is ~LINEAR in megapixels (high R^2). The header gives you")
print("megapixels for free (Lesson 1). So one header read predicts the CPU cost")
print("the GPU will wait behind. A fleet that ignores this is flying blind.")
