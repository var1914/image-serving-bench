# RunPod Runbook

Reproduce the whole suite on a single GPU pod. Times/prices are order-of-magnitude.

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
    watch -n1 'ss -tan | grep :8000 | wc -l'   # in-flight / queue proxy

## 8. Cost hygiene
- Stop the pod between sessions; keep the volume. Log GPU-hours in results/COST_LOG.md.
- RQ5 + RQ1-CPU can be prototyped on the MacBook before renting anything.

## 9. Pull results back
    tar czf results.tgz results/ && <download or push>   # then plot locally in analysis/
