#!/usr/bin/env python3
"""Stage split with a REAL model: decode vs preprocess vs ONNX inference (CPU).

A) per-image cost of each stage as image size grows (batch=1)
B) batching curve: how per-image inference cost falls as batch size grows
C) ORT intra-op threads: how much ONNX Runtime parallelises one inference itself
Content is photo-like (not smooth gradients) to avoid the known cheap-content bias.
"""
import io, os, sys, time, statistics
import numpy as np
from PIL import Image
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness"))
from cpuinfo import describe, effective_cores

THREADS = int(os.environ.get("PW_ORT_THREADS", effective_cores()))   # sized to the container

MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "repo", "resnet50_onnx", "1", "model.onnx")
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
rng = np.random.default_rng(0)

def photo_like_jpeg(mp):
    s = int((mp * 1e6) ** 0.5); k = max(8, s // 16)
    small = rng.integers(0, 256, (k, k, 3), dtype="uint8")
    img = Image.fromarray(small, "RGB").resize((s, s), Image.BICUBIC)
    b = io.BytesIO(); img.save(b, "JPEG", quality=85); return b.getvalue()

def med(fn, reps=7):
    fn(); ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts)

def decode(b):  im = Image.open(io.BytesIO(b)).convert("RGB"); im.load(); return im
def preprocess(im):
    x = np.asarray(im.resize((224, 224), Image.BILINEAR), np.float32) / 255.0
    return ((x - MEAN) / STD).transpose(2, 0, 1)[None]          # HWC -> NCHW, batch 1

def session(threads=0):
    so = ort.SessionOptions(); so.intra_op_num_threads = threads   # 0 = ORT default
    return ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])

sess = session(THREADS)
print(f"ONNX Runtime {ort.__version__}, CPU EP, intra-op threads={THREADS}")
print(f"CPU: {describe()}\n")

print("A) per-image stage cost, batch=1")
print(f"{'MP':>5} {'decode':>8} {'preproc':>8} {'infer':>8} {'total':>8}  {'decode share':>12}")
for mp in [0.3, 1, 3, 8, 12, 24]:
    b = photo_like_jpeg(mp); im = decode(b); x = preprocess(im)
    d = med(lambda: decode(b), 5); p = med(lambda: preprocess(im)); i = med(lambda: sess.run(None, {"input": x}))
    tot = d + p + i
    print(f"{mp:5.1f} {d:7.1f}ms {p:7.1f}ms {i:7.1f}ms {tot:7.1f}ms  {100*(d+p)/tot:10.0f}% pre")

print("\nB) batching curve (inference only)")
print(f"{'batch':>5} {'total':>9} {'per-image':>10} {'img/s':>8}")
x1 = preprocess(decode(photo_like_jpeg(1)))
for bs in [1, 2, 4, 8, 16, 32]:
    xb = np.repeat(x1, bs, axis=0)
    t = med(lambda: sess.run(None, {"input": xb}), 5)
    print(f"{bs:5d} {t:8.1f}ms {t/bs:9.1f}ms {1000*bs/t:8.0f}")

print("\nC) ORT intra-op threads (batch=1 and batch=8)")
print(f"{'threads':>7} {'b=1':>8} {'b=8 per-img':>12}")
x8 = np.repeat(x1, 8, axis=0)
for th in [1, 2, 4, 8, 0]:                   # 0 = ORT default: one thread per HOST core
    s = session(th)
    label = "default" if th == 0 else str(th)
    print(f"{label:>7} {med(lambda: s.run(None, {'input': x1}), 5):7.1f}ms {med(lambda: s.run(None, {'input': x8}), 5)/8:11.1f}ms")
