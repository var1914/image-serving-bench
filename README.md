# The Payload Is the Workload

**Benchmarking image-inference platforms under input heterogeneity.**

A vision service's per-request cost is set almost entirely by properties of the
*incoming bytes* — resolution, format, chroma subsampling, progressive encoding,
ICC profile — not by the model. Yet every control-plane layer (autoscaler
metrics, batchers, admission control, capacity models, and MLPerf Inference
itself) treats requests as interchangeable. This project measures how wrong that
assumption is, and shows the cheap fixes.

MLPerf hands us the opening: its rules make approved preprocessing **untimed** —
the industry-standard benchmark measures a world where JPEGs arrive
pre-decoded. Real services get bytes over HTTP.

## Research questions

- **RQ1 — Dispersion.** Fixed model, realistic image population: how large is the
  per-request cost spread from payload alone? (Metric: CDR = p99 cost / p50 cost,
  stage-decomposed.) Claim: >10×, larger than swapping the model.
- **RQ2 — Control-plane blindness.** Same RPS, different payload *mix* into an
  HPA/KEDA deployment: do CPU% / RPS signals stay flat while p99 catches fire?
  Which scaling signal actually tracks cost?
- **RQ3 — Header-only cost prediction.** JPEG SOF / PNG IHDR / WebP VP8X give
  width×height in ~µs with no decode. Can (bytes, pixels, format, progressive)
  predict cost well enough to drive a shape-aware autoscaler, admission
  controller, and shape-partitioned pool?
- **RQ4 — Hostile-but-valid payloads.** Pixel bombs, multi-scan progressive
  JPEG, animated WebP, CMYK+ICC, 16-bit PNG, truncated streams. (Metric: LAF =
  latency amplification factor.) A 40 KB file that costs like 40 MB. Price each
  defense. *Defensive framing: hardening your own endpoint.*
- **RQ5 — Decoder divergence.** Same image through Pillow / Pillow-SIMD /
  OpenCV / torchvision / libvips / nvJPEG: top-1 flip rate. An infra choice that
  produces a correctness bug.

## Deliverables (what a company can use the same afternoon)

1. An open load-harness they point at their own endpoint.
2. A CPU:GPU sizing formula for image serving.
3. A specific autoscaler metric to switch to (megapixels/s, not CPU%/RPS).
4. A hostile-payload hardening checklist with per-defense cost.

## Layout

    EXPERIMENT_PLAN.md   the plan — start here
    RUNBOOK_RUNPOD.md    step-by-step to reproduce on a RunPod GPU pod
    corpus/              payload population + hostile-payload generators
    sut/                 systems under test (fastapi_naive, triton)
    harness/             open-loop load generator + metrics collection
    analysis/            metric definitions + plotting
    results/             raw runs (gitignored except .gitkeep)

## Quickstart

```bash
# 1. install (Python 3.11+)
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# 2. start the server under test (SUT-A)
.venv/bin/python -m uvicorn sut.fastapi_naive.server:app --host 0.0.0.0 --port 8099

# 3. drive open-loop load against it
.venv/bin/python harness/loadgen.py --rps 40 --duration 60

# 4. run the convoy-tax experiment (payload variance vs tail latency)
.venv/bin/python harness/convoy_experiment.py --rps 60 --duration 90

# 5. (optional) live dashboards — Prometheus :9090, Grafana :3000 (auto-provisioned)
cd sut/observability && docker compose up -d
```

Metric definitions (CDR, LAF, goodput@SLO, the coordinated-omission check) are in
`analysis/metrics_defs.md`. See `RUNBOOK_RUNPOD.md` to reproduce at scale on a GPU pod.

## Status

Active research. Target venue: EuroMLSys / MLSys workshop (6-page), arXiv preprint.
