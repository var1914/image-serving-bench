# RunPod Runbook

Reproduce the whole suite on a single GPU pod. Times/prices are order-of-magnitude.

> **Which path to run:** for the **convoy-tax experiment (the current, tested one),
> follow §10 below** — it is self-contained and uses port **8099**. Sections 1–9 are
> the broader plan for later experiments (corpus generation, RQ1/RQ2/RQ5); some of
> their commands are placeholders with example values and are not runnable yet. When
> §1–9 and §10 disagree, **§10 wins**.

## 1. Pick the pod
- **Primary:** A40 48 GB, ~9 vCPU, 50 GB RAM, ~$0.4–0.8/hr. CPU-lean on purpose.
- **Container:** `runpod/pytorch:2.x-cuda12.x` (has CUDA + torch).
- **Volume:** 60 GB persistent at `/workspace` (corpus + results survive stops).
- Expose an HTTP port if you drive the SUT from your laptop; otherwise keep the
  generator on-pod (loopback) for the clean compute story (see plan §7).

## 2. First-boot setup
    cd /workspace && git clone <this-repo> pw && cd pw/payload_workload
    pip install -r requirements.txt        # pillow-simd, opencv, pyvips, nvidia-dali, torch, fastapi, uvicorn, pyarrow, pandas, numpy
    apt-get update && apt-get install -y libvips taskset linux-tools-common  # pyvips + core pinning + perf
    nvidia-smi ; nproc ; python -c "import torch;print(torch.cuda.get_device_name())"

## 3. Record the environment (goes in every results dir)
    scripts/env_snapshot.sh > results/$RUN/env.txt   # nproc, nvidia-smi, clocks, pod SKU, governor

## 4. Stabilize the box (control confounders)
    nvidia-smi -pm 1
    nvidia-smi -lgc <base>,<base>          # lock GPU clocks; log temps during run
    export DECODE_CORES=0-3                 # pin decode workers; taskset -c $DECODE_CORES ...

## 5. Build the corpus (once; regenerate from seed, don't ship images)
    python corpus/generate_corpus.py --population --n 50000 --seed 20260820 --out /workspace/corpus_P
    python corpus/generate_corpus.py --hostile    --out /workspace/corpus_H     # RQ4, defensive set

## 6. Run experiments (each writes results/<run>/)
    # RQ5 first — no server, fastest signal
    python analysis/rq5_decoder_divergence.py --corpus /workspace/corpus_P --out results/rq5

    # RQ1–RQ4: start a SUT, then drive it open-loop
    python sut/fastapi_naive/server.py &                        # or docker compose up (triton)
    python harness/loadgen.py --target http://localhost:8000/predict \
        --corpus /workspace/corpus_P --pattern poisson --rps 20 --duration 300 \
        --out results/rq1_suta
    # RQ2 mix-sweep, RQ3 predictor, RQ4 hostile: same loadgen, different --scenario flags

## 7. Watch it break (second terminal)
    nvidia-smi dmon -s u              # GPU SM% — watch it idle while queue grows (RQ2)
    py-spy top --pid <server pid>     # where CPU time goes (decode/resize dominate)
    watch -n1 'ss -tan | grep :8099 | wc -l'   # in-flight / queue proxy

## 8. Cost hygiene
- Stop the pod between sessions; keep the volume. Log GPU-hours in results/COST_LOG.md.
- RQ5 + RQ1-CPU can be prototyped on the MacBook before renting anything.

## 9. Pull results back
    tar czf results.tgz results/ && <download or push>   # then plot locally in analysis/

## 10. Convoy-tax run with CPU core pinning (clean tail measurement)

Goal: measure whether payload-size *variance* inflates the tail at constant average
load, and whether the server tracks the queueing-theory (Kingman) prediction.

Why pinning: the load generator must not compete with the server for CPU, or it
falls behind schedule and the tail latency is understated (coordinated omission).
Linux `taskset` isolates them onto different cores. This experiment is CPU-bound
(image decode) and does **not** use the GPU — the pod is chosen for its cores,
core-pinning, and clean Docker, not for the GPU.

Assume a pod with >= 9 vCPU (cores 0-8). Adjust core ranges to your pod.

    # 1. clone + install (skip the venv on a throwaway pod — base python already has pip;
    #    a container's `python -m venv` often ships WITHOUT pip, which just wastes time)
    git clone git@github.com:var1914/image-serving-bench.git && cd image-serving-bench
    python3 -m pip install -r requirements.txt
    #   (if pip itself is missing: apt-get update && apt-get install -y python3-pip)
    apt-get update && apt-get install -y util-linux htop      # taskset + htop

    # 2. server pinned to cores 0-6 (7 cores)
    taskset -c 0-6 python3 -m uvicorn sut.fastapi_naive.server:app \
        --host 0.0.0.0 --port 8099 --log-level warning &
    curl -s localhost:8099/healthz       # sanity: -> {"ok":true,...}

    # 3. live metrics — NO Docker needed (the pod is itself a container, so
    #    docker-in-docker for Grafana is painful; use the terminal dashboard instead)
    python3 harness/metrics_watch.py     # req/s, in-flight, p50/p99, mean MP — live
    #   in another shell:  htop          # CPU per core (server 0-6 busy, client 7-8)
    #   (only if your pod actually supports Docker: cd sut/observability && docker compose up -d)

    # 4. experiment pinned to cores 7-8, warmup on, utilization proxy on
    taskset -c 7-8 python3 harness/convoy_experiment.py \
        --rps 700 --duration 90 --warmup 20 --server-cores 7 \
        --max-conns 512 --out results/convoy

### Tuning the load (the one ops knob)
The script prints an `operating point: ... rho X.XX [OK|TOO LOW|TOO HIGH]` line.
Adjust `--rps` until `rho` is ~0.7-0.9 **and** C0 goodput stays ~100%:
- capacity ~= server_cores / (service_ms/1000); with ~7.3 ms/req on 7 cores that is
  ~960 req/s, so rho 0.8 is around 700-780 rps. Real per-request time on the pod may
  differ, so trust the printed rho and htop over the arithmetic.
- Cross-check in a second terminal: `htop` (cores 0-6 busy ~70-90%, cores 7-8 carry
  the client), plus the Grafana p99 + payload-megapixels panels.

### Reading the result
- Each row prints measured tax vs Kingman, plus `achieved/offered rps` and `om_p99`.
- The footer prints a trustworthiness gate: `rho in band?` and `omission low?`.
  If any row shows high `om_p99` (generator fell behind) or rho is out of band, the
  numbers are not trustworthy — retune `--rps` / raise `--max-conns` and rerun.
- Full data in `results/convoy/convoy_results.json`. Pull it back:
      tar czf convoy.tgz results/convoy && <download>
