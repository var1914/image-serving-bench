# Metric definitions (single source of truth)

All plots and claims trace to these. Per-request rows come from the server
(stage timings, joined by id) + the loadgen (send schedule, omission, latency).

- **Stage vector** `L = (t_decode, t_resize, t_norm, t_h2d, t_compute, t_d2h)`.
  Server measures each around the real call; sum + queue wait = total.
- **CDR** (Cost Dispersion Ratio) = `p99(cost) / p50(cost)`, per stage and total.
  cost = CPU-side work (decode+resize+norm) so it is load-independent. RQ1.
- **LAF** (Latency Amplification Factor) = `cost(hostile) / cost(benign twin of
  equal on-disk bytes)`. Per hostile class. RQ4.
- **MPPS** = megapixels decoded / (wall-seconds x pinned cores). Portable unit;
  the headline capacity number that survives across pod SKUs.
- **Goodput@SLO** = `served_under_SLO / offered`, at a fixed offered rate. SLO
  set once (e.g. p99 < 150 ms for ResNet-50 @ 224). RQ2/RQ3/RQ4.
- **Signal fidelity** = Pearson/Spearman corr(signal_t, true_p99_t) across a
  mix sweep, for signal in {CPU%, RPS, in-flight, queue-depth, MPPS}. RQ2/RQ3.
- **GPU idle-under-load** = mean GPU SM-active% while queue-depth > 0. Proves the
  GPU starves behind the CPU decode wall. RQ2.
- **Flip rate** = fraction of images whose top-1 label differs across two decoder
  backends, weights fixed. Pairwise matrix. RQ5.
- **Coordinated-omission delta** = `actual_send - scheduled_send` (loadgen). A
  nonzero p99 here means the generator itself fell behind => trust the tail only
  when this is near zero, else raise concurrency / lower rps.

Reporting rule: >=5 runs/cell, randomized corpus order, median + p95 bootstrap CI,
discard first 60 s warmup. A number without a CI is a claim without evidence.
