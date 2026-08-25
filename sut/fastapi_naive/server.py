#!/usr/bin/env python3
"""SUT-A: the naive image-serving stack + concurrency footgun + Prometheus metrics.

CONCURRENCY: async def is COOPERATIVE, not parallel. A coroutine owns the
single event-loop thread until it hits `await`; call a blocking function directly
and every other request + health check freezes behind you.
  /predict_blocking   async def -> blocking work ON the event loop   (BUG)
  /predict_offloaded  async def -> work handed to a thread pool       (FIX)
  /healthz            trivial canary
  /metrics            Prometheus scrape endpoint

OBSERVABILITY: every request is measured. Prometheus scrapes /metrics;
Grafana graphs it. The metric that matters for THIS paper is pw_payload_megapixels
sitting right next to pw_request_duration_seconds -- so you can watch cost track
the payload, live.

Run:  .venv/bin/python -m uvicorn sut.fastapi_naive.server:app --host 0.0.0.0 --port 8099
      (from the payload_workload/ dir so the import path resolves)
"""
from __future__ import annotations
import asyncio, io, time
from fastapi import FastAPI, Request, Response
from PIL import Image
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI(title="SUT-A naive image server")

# ---- metrics -----------------------------------------------------------------
# Histograms bucket observations so Prometheus can compute quantiles (p50/p99)
# across the whole fleet with histogram_quantile(). Buckets are in SECONDS.
LAT_BUCKETS   = (.005,.01,.02,.05,.1,.15,.2,.3,.5,.75,1,2,5,10,30)  # 0.15 = our SLO
STAGE_BUCKETS = (.001,.005,.01,.02,.05,.1,.2,.5,1,2)
MP_BUCKETS    = (.1,.5,1,2,4,8,12,24,48)

REQS     = Counter("pw_requests_total", "requests", ["path", "code"])
LAT      = Histogram("pw_request_duration_seconds", "end-to-end latency", ["path"], buckets=LAT_BUCKETS)
INFLIGHT = Gauge("pw_inflight", "in-flight requests (queue-depth proxy)")
DECODE   = Histogram("pw_decode_seconds", "decode stage", buckets=STAGE_BUCKETS)
RESIZE   = Histogram("pw_resize_seconds", "resize stage", buckets=STAGE_BUCKETS)
MP       = Histogram("pw_payload_megapixels", "payload size (the cost driver)", buckets=MP_BUCKETS)

@app.get("/metrics")                      # exact path, no redirect -> clean for Prometheus
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


def decode_resize(body: bytes) -> dict:
    """The real, blocking CPU work every image request does before the GPU."""
    t0 = time.perf_counter()
    img = Image.open(io.BytesIO(body)).convert("RGB")
    img.load()                                    # force full decode (PIL is lazy)
    t1 = time.perf_counter()
    megapixels = img.width * img.height / 1e6     # the cost driver, from the pixels
    img.resize((224, 224), Image.BILINEAR)
    t2 = time.perf_counter()
    return {"decode_s": t1 - t0, "resize_s": t2 - t1, "megapixels": megapixels}


def observe(stages: dict) -> dict:
    DECODE.observe(stages["decode_s"]); RESIZE.observe(stages["resize_s"])
    MP.observe(stages["megapixels"])
    return {"decode_ms": round(stages["decode_s"] * 1e3, 2),
            "resize_ms": round(stages["resize_s"] * 1e3, 2),
            "megapixels": round(stages["megapixels"], 2)}


@app.get("/healthz")
async def healthz():
    return {"ok": True, "t": time.perf_counter()}


@app.post("/predict_blocking")
async def predict_blocking(request: Request):
    """THE BUG: decode_resize() runs on the event-loop thread -> freezes everyone."""
    body = await request.body()
    stages = decode_resize(body)                  # blocks the ONLY event-loop thread
    return {"path": "blocking", **observe(stages)}


@app.post("/predict_offloaded")
async def predict_offloaded(request: Request):
    """THE FIX: same work in the default thread pool -> loop stays free."""
    body = await request.body()
    loop = asyncio.get_running_loop()
    stages = await loop.run_in_executor(None, decode_resize, body)
    return {"path": "offloaded", **observe(stages)}
