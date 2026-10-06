#!/usr/bin/env python3
"""SUT-C: image classification with a REAL model — ResNet-50 in ONNX Runtime (CPU).

Request path: bytes -> decode (Pillow) -> resize 224 + normalize (numpy) -> ONNX inference.
All three stages run in a bounded worker pool so the event loop stays free.

Two knobs, which together decide how the CPU is shared:
  PW_WORKERS      how many requests are processed at the same time (request-level parallelism)
  PW_ORT_THREADS  how many threads ONE inference may use inside ONNX Runtime
                  (intra-op parallelism; 0 = ONNX Runtime's default, ~all cores)

  (a) PW_WORKERS=1  PW_ORT_THREADS=0   one request at a time, inference uses every core
  (b) PW_WORKERS=N  PW_ORT_THREADS=1   N requests at once, one thread each
  (c) PW_WORKERS=N  PW_ORT_THREADS=0   N requests at once AND each grabs every core
                                        -> N x cores threads fighting over the same cores

Run (from payload_workload/):
  PW_WORKERS=4 PW_ORT_THREADS=1 python -m uvicorn sut.onnx_server.server:app --port 8100
"""
from __future__ import annotations
import asyncio, io, os, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, Request, Response
from PIL import Image
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("PW_MODEL", os.path.join(HERE, "..", "..", "models", "repo",
                                                "resnet50_onnx", "1", "model.onnx"))
from harness.cpuinfo import describe, effective_cores

WORKERS = int(os.environ.get("PW_WORKERS", effective_cores()))
ORT_THREADS = int(os.environ.get("PW_ORT_THREADS", "0"))   # 0 = ORT default (host-sized!)
ORT_SPIN = os.environ.get("PW_ORT_SPIN", "1")             # "0" = threads sleep instead of spin-waiting

so = ort.SessionOptions()
so.intra_op_num_threads = ORT_THREADS
so.add_session_config_entry("session.intra_op.allow_spinning", ORT_SPIN)
SESSION = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
INPUT = SESSION.get_inputs()[0].name
POOL = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="req")
print(f"[SUT-C] ResNet-50 ONNX | workers={WORKERS} ort_threads={ORT_THREADS or 'default'} "
      f"spin={ORT_SPIN} | {describe()}", flush=True)

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

app = FastAPI(title="SUT-C ONNX ResNet-50")

LAT_BUCKETS = (.005, .01, .02, .05, .1, .15, .2, .3, .5, .75, 1, 2, 5, 10, 30)
STAGE_BUCKETS = (.001, .005, .01, .02, .05, .1, .2, .5, 1, 2)
MP_BUCKETS = (.1, .5, 1, 2, 4, 8, 12, 24, 48)
REQS = Counter("pw_requests_total", "requests", ["path", "code"])
LAT = Histogram("pw_request_duration_seconds", "end-to-end latency", ["path"], buckets=LAT_BUCKETS)
INFLIGHT = Gauge("pw_inflight", "in-flight requests (queue-depth proxy)")
STAGE = Histogram("pw_stage_seconds", "per-stage time", ["stage"], buckets=STAGE_BUCKETS)
MP = Histogram("pw_payload_megapixels", "payload size", buckets=MP_BUCKETS)


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.middleware("http")
async def instrument(request: Request, call_next):
    path = request.url.path
    if path == "/metrics":
        return await call_next(request)
    INFLIGHT.inc()
    start = time.perf_counter()
    code = "500"
    try:
        resp = await call_next(request)
        code = str(resp.status_code)
        return resp
    finally:
        INFLIGHT.dec()
        LAT.labels(path).observe(time.perf_counter() - start)
        REQS.labels(path, code).inc()


def pipeline(body: bytes) -> dict:
    t0 = time.perf_counter()
    img = Image.open(io.BytesIO(body)).convert("RGB")
    img.load()
    mp = img.width * img.height / 1e6
    t1 = time.perf_counter()
    x = np.asarray(img.resize((224, 224), Image.BILINEAR), np.float32) / 255.0
    x = ((x - MEAN) / STD).transpose(2, 0, 1)[None]
    t2 = time.perf_counter()
    logits = SESSION.run(None, {INPUT: x})[0]
    t3 = time.perf_counter()
    STAGE.labels("decode").observe(t1 - t0)
    STAGE.labels("preprocess").observe(t2 - t1)
    STAGE.labels("infer").observe(t3 - t2)
    MP.observe(mp)
    return {"decode_ms": round((t1 - t0) * 1e3, 2), "preprocess_ms": round((t2 - t1) * 1e3, 2),
            "infer_ms": round((t3 - t2) * 1e3, 2), "megapixels": round(mp, 2),
            "top1": int(logits.argmax())}


@app.get("/healthz")
async def healthz():
    return {"ok": True, "workers": WORKERS, "ort_threads": ORT_THREADS}


@app.post("/predict")
async def predict(request: Request):
    body = await request.body()
    return await asyncio.get_running_loop().run_in_executor(POOL, pipeline, body)
