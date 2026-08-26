# The Payload Is the Workload

**Cost-aware serving for image inference under input heterogeneity.**

In a vision API, per-request cost is dominated by the *payload* — megapixels,
format, chroma subsampling — not by the model. That preprocessing dominates
serving latency is already established ([AbouElhamayed et al., DAC 2024](https://arxiv.org/abs/2403.12981)).
This project asks the next question: **when the payload *mix* shifts, the control
plane is blind to it — and what cheap signal fixes that?**

Autoscalers (on CPU% / GPU-utilization / RPS), count-based batchers, and capacity
models all treat requests as interchangeable. Under a payload-mix shift those
signals stay flat while the tail catches fire. We measure that failure and test a
fix: a **header-only cost signal** — width×height read from the image header in
microseconds, *no decode* — used to drive autoscaling and admission.

## What this is (and isn't)
- **Is:** an open harness + experiments on *heterogeneous, open-loop* image
  payloads, and a cheap input-derived cost signal for the control plane.
- **Isn't:** a rediscovery that preprocessing is expensive (that's the baseline
  below), nor a new load generator (MLPerf LoadGen already does open-loop Poisson).

## Contributions under test
- **C1 (headline) — Header-only cost signal for autoscaling.** Parse
  width×height/format from header bytes (no decode); predict per-request CPU cost;
  scale/admit on megapixels-per-second. Claim: holds SLO through a payload-mix
  shift where CPU%- / utilization-based scaling fails.
- **C2 (mechanism) — Deviation from queueing theory.** At equal mean load, how
  much does payload-size *variance* inflate the tail in a real Python stack
  (GIL + event loop + thread pool), and does Kingman's G/G/1 bound predict it?
  The *deviation* — not the direction — is the result.
- **C3 (supporting) — Decode-stage cost amplification.** A valid image with a
  huge pixel count in tiny bytes hits the *decoder* (before the fixed-size resize);
  we report the latency-amplification factor and show utilization-based autoscalers
  *amplify* rather than absorb it.

## Related work / positioning
- **[Beyond Inference (AbouElhamayed et al., DAC 2024)](https://arxiv.org/abs/2403.12981)** —
  preprocessing dominates DNN-serving latency; homogeneous payloads, closed-loop.
  *Our baseline.* We move to heterogeneous payloads + open-loop arrivals and add a
  control-plane fix.
- **[MLPerf LoadGen (arXiv:1911.02549)](https://arxiv.org/pdf/1911.02549)** — the
  Poisson, open-loop, coordinated-omission-safe generator. *We follow its
  methodology*; our generator is a lightweight stand-in for local runs.
- **[Size-aware Sharding (Didona & Zwaenepoel, NSDI'19)](https://www.usenix.org/conference/nsdi19/presentation/didona)** —
  request-size heterogeneity causes head-of-line blocking / tail inflation in
  key-value stores. *Different substrate* (image decode + Python/GIL); we add a
  header-derived predictor as the control signal, not just size-aware routing.
- **[Sponge Examples (Shumailov et al., EuroS&P 2021)](https://arxiv.org/abs/2006.03463)** —
  energy-latency DoS on the *model* at fixed input size. *We target the decode
  stage before resize* and frame it as measurement, not a new attack.

## Quickstart

```bash
# install (Python 3.11+); on a throwaway box you can skip the venv and use base python
python3 -m pip install -r requirements.txt

# start the server under test (SUT-A: FastAPI + Prometheus /metrics)
python3 -m uvicorn sut.fastapi_naive.server:app --host 0.0.0.0 --port 8099

# drive open-loop load against it
python3 harness/loadgen.py --rps 40 --duration 60

# the convoy experiment: payload variance vs tail latency, measured vs Kingman
python3 harness/convoy_experiment.py --rps 700 --duration 90 --warmup 20 --server-cores 7

# live metrics in the terminal — no Docker needed
python3 harness/metrics_watch.py
```

Metric definitions (CDR, LAF, goodput@SLO, the coordinated-omission check) are in
`analysis/metrics_defs.md`. `RUNBOOK_RUNPOD.md` §10 has the core-pinned setup for a
clean tail measurement on a GPU pod.

## Layout

    EXPERIMENT_PLAN.md   design + positioning
    RUNBOOK_RUNPOD.md    reproduce on a GPU pod (see §10 for the pinned convoy run)
    sut/                 system under test (fastapi_naive) + observability stack
    harness/             open-loop load generator, convoy experiment, metrics watcher
    corpus/              header probe + payload generators
    analysis/            metric definitions + mechanism demos
    results/             raw runs (gitignored except .gitkeep)

## Status

Active research; scope is being pressure-tested against the prior work above.
Target: EuroMLSys / MLSys workshop (6-page) + arXiv — contingent on a wedge that
survives a full related-work sweep.
