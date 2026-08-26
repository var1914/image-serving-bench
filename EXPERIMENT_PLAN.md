# Experiment Plan — The Payload Is the Workload

## Positioning (updated after a prior-art check)

Honest scoping: some early framings here are already published and are treated as
**baselines**, not contributions.

- **Preprocessing dominates serving latency** — TAKEN by
  [Beyond Inference, DAC 2024](https://arxiv.org/abs/2403.12981) (up to 97% for
  large images; CPU:GPU sizing; multi-GPU stall). RQ1 (dispersion) and the
  CPU:GPU-sizing artifact are *confirmation of a known result*, not new.
- **Open-loop / coordinated-omission-safe load** — TAKEN by
  [MLPerf LoadGen](https://arxiv.org/pdf/1911.02549). Our generator follows it.
- **Variance/size heterogeneity inflates the tail (direction)** — TAKEN by
  [Size-aware Sharding, NSDI'19](https://www.usenix.org/conference/nsdi19/presentation/didona)
  (HOL blocking in KV stores). "Does variance hurt the tail?" is not a question.
- **Energy-latency DoS on the model** — TAKEN by
  [Sponge Examples, EuroS&P 2021](https://arxiv.org/abs/2006.03463) (fixed input
  size; attacks the model, not the decoder).

Defensible contributions the experiments below now serve (pending a full venue sweep):

- **C1 (headline) — header-only cost signal for autoscaling.** Predict per-request
  cost from header bytes (no decode); scale/admit on megapixels/s; hold SLO through
  a payload-mix shift where CPU%- / util-based scaling fails. *(was RQ3.)*
- **C2 (mechanism) — deviation from queueing theory.** How far a real Python
  (GIL + event loop + thread pool) image server departs from Kingman's G/G/1 tail
  bound under heterogeneous, open-loop payloads — the *deviation*, not the
  direction. *(was RQ2 + the convoy experiment.)*
- **C3 (supporting) — decode-stage cost amplification** (before the resize) and the
  fact that util-based autoscalers amplify rather than absorb it. *(was RQ4.)*

Read the RQ1–RQ5 sections below through the C1/C2/C3 lens; RQ1 and RQ5 are demoted
to confirmation/appendix.

---

Goal: a benchmark/stress-suite that **provably breaks** production image-serving
assumptions under input heterogeneity, at scale, on a real GPU. Every experiment
has a stated hypothesis, a defined "break" (the failure we force into the open),
a scale target, and a pass/fail bar so a negative result is still publishable.

---

## 0. The one-sentence thesis (and why it breaks things)

> In an image service the GPU is rarely the bottleneck; the **CPU decode+resize
> path is**, and its cost is a heavy-tailed function of bytes the caller
> controls. So a fleet sized on average cost, scaled on CPU%/RPS, and batched by
> count will fall over when the *mix* of payloads shifts — with no change in
> request rate and no error the control plane can see.

Everything below is engineered to make that sentence visible on a plot.

---

## 1. Hardware target (RunPod) — and why the CPU:GPU ratio *is* the paper

Preprocessing is CPU-bound; inference is GPU-bound. RunPod pods ship a fixed
vCPU allotment per GPU, and that ratio is exactly the knob under test.

**Primary pod:** 1× **NVIDIA A40 (48 GB)**, ~9 vCPU, ~50 GB RAM.
- Cheap (~$0.40–0.79/hr), plentiful, representative of a mid-tier serving box.
- Deliberately CPU-lean so the decode wall shows up early — that is a feature.

**Secondary pod (contrast):** 1× **L40S** or **A100 80 GB**, higher vCPU.
- Used only for RQ1/RQ2 to show the break *moves* but does not disappear as you
  buy a bigger GPU — you cannot GPU your way out of a CPU-preprocessing wall.

**Control the confounder:** pin decode workers to N physical cores with
`taskset`/cgroup cpuset; record `nproc`, governor, and whether the pod is
shared. Report all latencies as (pod SKU, pinned cores) tuples. The headline
number is **megapixels/s/core**, which is portable across pods.

GPU-hour budget: RQ1–RQ4 need ~6–10 A40-hours total; RQ5 is CPU+GPU decode only.
Keep a running tab in `results/COST_LOG.md`.

---

## 2. Systems under test (SUT)

Two configs bracket what companies actually run. Same model weights, same
target resolution, same SLO — only the platform differs.

**SUT-A — `fastapi_naive` (the strawman most startups run).**
- FastAPI + uvicorn, `Pillow.open().convert("RGB").resize()` on the request
  thread, single model, `torch` on GPU, no batching.
- Sync decode in the event loop path (the classic latent bug) *and* a
  threadpool variant, both measured.

**SUT-B — `triton` (the "did it right" config).**
- Triton Inference Server, DALI or nvJPEG GPU decode, dynamic batching,
  multiple model instances, TensorRT engine.
- This is the config vendors demo. We show even *this* breaks on RQ2/RQ4 unless
  it is made shape-aware.

**Model anchors.**
- **ResNet-50** — MLPerf-comparable anchor; keeps us honest vs. published numbers.
- **ViT-B/16** — modern, and the vehicle for RQ5 flip-rate (interpolation
  sensitivity differs from CNNs).
- (Optional) **YOLOv8** — detection is resolution-sensitive; strengthens RQ1/RQ4.

Fixed across SUTs: target 224×224 (ResNet/ViT), bilinear, ImageNet normalize,
same class labels — so any output difference in RQ5 is *purely* the decoder.

---

## 3. Corpus design (the independent variable)

Two corpora. Both generated deterministically (seeded) by `corpus/generate_corpus.py`
so runs are reproducible; store only the generator + a manifest, not 50 GB of images.

### 3a. Realistic population `P` (RQ1–RQ3)
Real user uploads are heavy-tailed, not ImageNet-uniform. Draw ~50k images:
- **Resolution:** megapixels ~ log-normal, median ~1.2 MP, p99 ~24 MP, capped at
  a plausible phone-camera 48 MP. (This tail is the whole point.)
- **Format mix:** JPEG 70% / PNG 15% / WebP 15%.
- **JPEG chroma:** 4:2:0 60%, 4:2:2 25%, 4:4:4 15%; quality 60–95.
- **Progressive JPEG:** 20% of JPEGs (multi-scan decode is ~2–3× baseline).
- **Color:** 10% carry a non-sRGB ICC profile; 5% CMYK; EXIF orientation on 30%.
- Content: mix of real photos (COCO/OpenImages sample) + synthetic gradients so
  entropy/quality is controlled.

Manifest columns (per image): `id, format, width, height, megapixels, bytes,
subsampling, progressive, icc, cmyk, exif_orient, jpeg_quality, sha256`. This is
the ground truth for RQ3 regression.

### 3b. Hostile-but-valid corpus `H` (RQ4) — defensive hardening set
Every file is a *valid* image that decodes to something legitimate; none exploit
a memory bug — they exploit the *cost model*. Small on disk, huge in cost.
- **Pixel bomb:** valid JPEG/PNG, e.g. 30000×30000 (0.9 GP) in a ~40 KB file.
- **Progressive-scan bomb:** JPEG with the max number of scans (worst-case
  multi-pass decode).
- **Animated WebP / APNG:** N frames; naive decoders decode all frames.
- **CMYK + fat ICC:** forces colorspace conversion + profile parse.
- **16-bit / high-bit-depth PNG:** doubles memory + conversion cost.
- **Deep-zoom tiled TIFF** (if backend accepts it): pathological tiling.
- **Truncated / lie-in-header:** dimensions in header ≠ actual scan (tests
  guard code paths; must fail cheap, not hang).
Each with a benign "twin" of equal on-disk bytes for the amplification baseline.

> Framing: this is authorized stress-testing of *your own* endpoint. The output
> is a hardening checklist, not an attack tool. No target but the SUT we run.

---

## 4. Metrics (formal)

Per request the harness records: recv→decode→resize→normalize→H2D→compute→D2H→send
timestamps, plus payload metadata joined from the manifest.

- **Stage latency vector** `L = (t_decode, t_resize, t_norm, t_h2d, t_compute, t_d2h)`.
- **CDR (Cost Dispersion Ratio)** = p99(total_cpu_cost) / p50(total_cpu_cost),
  reported per stage. Headline for RQ1.
- **LAF (Latency Amplification Factor)** = cost(hostile) / cost(benign twin of
  equal bytes). Headline for RQ4.
- **MPPS** = megapixels decoded per second per pinned core. Portable capacity unit.
- **Goodput@SLO** = requests served under SLO / offered, at fixed offered load.
- **Signal fidelity** = correlation between an autoscaler signal (CPU%, RPS,
  in-flight, queue depth, MPPS) and true p99 latency across a mix sweep. RQ2/RQ3.
- **Flip rate** = fraction of images whose top-1 label changes across decoder
  backends, holding weights fixed. RQ5.
- **GPU idle-under-load** = GPU SM-active% while the queue is non-empty (proves
  the GPU starves waiting on CPU decode).

---

## 5. Load generation methodology (get this right or the paper is wrong)

- **Open-loop, not closed-loop.** Requests are emitted on a schedule (Poisson or
  fixed-rate), *not* "send next after previous returns." Closed-loop hides tail
  latency (coordinated omission). We implement wrk2-style constant-arrival with
  omission correction. Locust/ab are closed-loop → not used for tail numbers.
- **Bodies are real bytes.** The generator POSTs actual image files (or references
  them for the server to fetch), so decode cost is real, not simulated.
- **Warm then measure.** Discard first 60 s (JIT, TF autotune, allocator warmup,
  CUDA context, TRT profile selection). Report steady state.
- **Saturation discipline.** For throughput, ramp offered RPS until goodput@SLO
  peaks then collapses; report the knee, not a single point.
- **Repeat & CI.** ≥5 runs per cell, report median + p95 bootstrap CI. A single
  run is an anecdote.

Generator lives in `harness/loadgen.py`; per-request rows to
`results/<run>/requests.parquet`; server-side stage timings to
`results/<run>/stages.parquet`; joined on request id.

---

## 6. The experiments

### RQ1 — Dispersion: payload spread > model spread
**Hypothesis.** On SUT-A, fixed ResNet-50, feeding population `P`, CDR ≥ 10×,
dominated by `t_decode` + `t_resize`, and the payload-induced spread exceeds the
p50 gap between ResNet-50 and ViT-B on fixed input.
**Method.** Replay `P` at low, non-saturating RPS (isolate cost, not queuing).
Record stage vectors. Compute CDR overall + per stage. Cross-plot cost vs.
megapixels/bytes/format.
**The break.** A latency histogram that is multi-modal by format and long-tailed
by megapixels — the "average request" is a fiction.
**Scale.** 50k images × ≥3 repeats × 2 SUTs × 2 models.
**Plot.** CDR bar per stage; cost-vs-megapixels hexbin colored by format.
**Pass/fail.** Publishable either way; CDR<3× would itself be a surprising null.

### RQ2 — Control-plane blindness: break the autoscaler without touching RPS
**Hypothesis.** Hold offered RPS fixed; shift the payload *mix* from p50 to p90
megapixels over a ramp. p99 latency and queue depth explode while CPU% and RPS —
the signals HPA/KEDA usually watch — stay ~flat, so the (simulated) autoscaler
never scales. GPU sits idle behind a CPU decode wall.
**Method.** Fixed 1× replica. Sweep mix in stages (all-small → mixed → all-large)
at constant arrival rate. Log every candidate signal + true p99 + GPU SM%.
Replay the same trace against a stand-in HPA controller reacting to CPU% vs. to
queue-depth vs. to MPPS; count SLO-violation seconds under each.
**The break.** A time series where the SLO is on fire and the CPU%/RPS lines are
flat — the autoscaler's blind spot, on one chart.
**Scale.** Multi-minute traces; ≥5 mix stages; both SUTs (show Triton's dynamic
batcher batches by *count* and mis-sizes when large images arrive).
**Plot.** Stacked timeline: offered RPS (flat) | CPU% (flat) | p99 (spiking) |
GPU-idle% | replicas-that-should-have-launched.
**Pass/fail.** Break confirmed if CPU%/RPS correlation with p99 < 0.3 while
queue-depth/MPPS correlation > 0.8.

### RQ3 — Header-only cost prediction: the cheap fix
**Hypothesis.** Parsing just the header (JPEG SOF0/2, PNG IHDR, WebP VP8X) —
microseconds, no full decode — yields features that predict per-request cost with
R² ≥ 0.85, enough to drive shape-aware scheduling.
**Method.** From the manifest, fit cost ~ f(pixels, bytes, format, progressive,
subsampling). Try linear, then gradient-boosted. Then *use* it three ways and
measure the win: (a) **MPPS-based autoscaler** vs. CPU%/RPS on the RQ2 trace;
(b) **admission control** — reject/route requests whose predicted cost > budget;
(c) **shape-partitioned pools** — route small vs. large payloads to separate
worker pools to kill head-of-line blocking.
**The break it fixes.** Re-run RQ2 with each mechanism; show goodput@SLO recovers.
**Scale.** Same 50k; 80/20 train/test split on the population.
**Plot.** Predicted-vs-actual cost; goodput@SLO before/after each mechanism.
**Pass/fail.** Success = ≥30% goodput@SLO recovery under mixed load from a
header-only predictor.

### RQ4 — Hostile-but-valid payloads: the asymmetric cost attack (and its price)
**Hypothesis.** Individual valid files achieve LAF ≥ 100× (a 40 KB pixel bomb
costs like a 40 MB image). A trickle of them (well under normal RPS) drops
goodput@SLO below 50%. Each standard defense (pixel-count guard, byte cap,
decode timeout, frame cap, format allowlist) has a measurable cost and a bypass.
**Method.** (i) Single-request LAF for each hostile class vs. its byte-twin.
(ii) Saturation: inject hostile requests at rising fraction of total load;
find the fraction that breaks SLO. (iii) Defense matrix: enable each guard, remit
the attack, measure residual LAF + false-positive rate on population `P` + added
per-request latency of the guard itself.
**The break.** A DoS curve: goodput vs. hostile-fraction, collapsing at a tiny
fraction. Then the same curve flattened by the guard stack.
**Scale.** ~8 hostile classes × benign twins; fraction sweep 0→5%.
**Plot.** LAF bar per class; goodput-vs-hostile-fraction before/after guards.
**Pass/fail.** Break confirmed if <2% hostile traffic pushes goodput@SLO <50%.
**Ethics/scope.** Only the SUT we run is targeted; no external systems; the
artifact is a defensive checklist. Generator emits files locally; nothing is
sent anywhere but localhost:SUT.

### RQ5 — Decoder divergence: an infra choice that becomes a correctness bug
**Hypothesis.** The *same* image + *same* weights yields different top-1 labels
across decode/resize backends (Pillow vs. Pillow-SIMD vs. OpenCV vs. torchvision
vs. libvips vs. nvJPEG), flip rate ≥ 1% on a hard subset, driven by
resize-kernel and chroma-upsampling differences, worse on ViT than ResNet.
**Method.** Decode/resize the same N images through each backend to float
tensors; measure pixel-level diff and end-to-end label agreement vs. a reference.
Isolate the cause: interpolation (bilinear impl differs), antialiasing on/off,
chroma upsampling, integer rounding, sRGB/gamma handling.
**The break.** "Which pod answered?" changes the prediction — a fleet with mixed
decoder versions is nondeterministic by deployment, not by design.
**Scale.** ImageNet val hard subset (low-margin logits) + population `P`.
**Plot.** Pairwise flip-rate matrix across backends; mean abs pixel diff.
**Pass/fail.** Break confirmed if any backend pair flips ≥1% of the hard subset.

---

## 7. Confounders & controls (reviewer defense)

- Thermal/clock drift → pin clocks (`nvidia-smi -lgc`), log temps, discard warmup.
- Noisy neighbor on shared pods → prefer dedicated; log `steal` time; repeat.
- Allocator/cache effects → randomize corpus order per run; report across runs.
- Network in the loop → co-locate generator on the pod (loopback) to isolate the
  compute story; a separate "over-the-wire" run quantifies transport for realism.
- Coordinated omission → open-loop generator (§5).
- Batcher interactions in Triton → sweep max_batch/queue-delay so we don't
  strawman it; report its best config.
- Model warm state → fixed engine, no lazy init in the measured window.

---

## 8. Roadmap

Infra spine first; image internals are a fixed input (cost ∝ megapixels; the
header exposes megapixels cheaply). CV specifics (chroma, ICC, decoder
divergence) are an optional appendix, not a prerequisite.

1. **SUT-A serving stack** + event-loop-blocking baseline (the async footgun).
2. **Serving mechanics** — the GIL & worker processes (threads vs processes for
   CPU-bound decode).
3. **Open-loop load generator + metrics** — percentiles, goodput@SLO, queue
   depth, coordinated-omission control.
4. **RQ2 — control-plane blindness** across a payload-mix sweep (which signal
   tracks cost: CPU%/RPS vs queue-depth/MPPS).
5. **RQ3 — cost-aware fix** — MPPS scaling signal, admission control,
   shape-partitioned pools; goodput before/after.
6. **RQ4 — resource-exhaustion hardening** — LAF, DoS curve, defense matrix.

Later / optional: RQ1 dispersion writeup; SUT-B (Triton) repeat; RQ5 decoder
divergence (CV appendix).

## 9. Risks & mitigations

- *GPU is not the bottleneck, so why rent one?* — precisely the finding; the GPU
  run proves GPU-idle-under-load and lets RQ2/RQ4 saturate realistically. RQ5 and
  RQ1-CPU can start on the laptop before spending a cent.
- *Triton setup eats a day* — SUT-A carries RQ1–RQ4; Triton is confirmation.
- *Corpus too big to store* — store generator + seed + manifest; regenerate.
- *Null result on RQ1* — a low CDR is itself a publishable surprise; framing holds.

## 10. What "done" looks like

Four reusable artifacts (harness, sizing formula, autoscaler metric, hardening
checklist) + a 6-page workshop paper + arXiv preprint. Every claim traces to a
seeded run in `results/` with a CI, not a single number.
