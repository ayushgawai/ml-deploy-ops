# ml-deploy-ops

Model deployment, evaluation and monitoring control plane. Register models, run
eval gates, canary promotion, latency/error metrics, CLI, tests.

> **Status (honest):** the code and tests are written, but at publish time the
> test suite and benchmark had **not yet been executed** in the build
> environment — every `python -m pytest` / benchmark invocation was blocked by
> the local tool-approval gate before any process ran (git operations were
> allowed; Python execution was not). The sanity-run section below is therefore
> marked *pending*, not filled with invented numbers. Run the commands under
> [Run / test / bench](#run--test--bench) to produce real output; treat any
> claim in this README as unverified until `pytest` is green locally.

## Design

```
                         +----------------------+
   register              |      Registry        |   JSON file (optional)
  ---------------------> |  ModelVersion, stage | <----------------------.
                         +----------+-----------+                        |
                                    | get / set_stage                  | persist
                         +----------v-----------+                        |
   metrics dict          |      EvalGate        |                        |
  ---------------------> | thresholds + baseline|                        |
                         | regression check     |                        |
                         +----------+-----------+                        |
                          pass |          | fail                         |
                               v          v REJECTED (cannot deploy)     |
                         +----------+-----------+                        |
                         |   CanaryController   |  traffic % (<=100)     |
                         | deploy/shift/promote |------------------------'
                         | rollback, auto-check |
                         +----------+-----------+
                                    | RequestMetric (latency, ok)
                         +----------v-----------+
                         |     MetricsStore     |  p50/p95/p99, error rate
                         |  summaries + alerts  |----------------> alerts /
                         +----------------------+                    auto-rollback

        ControlPlane = Registry + EvalGate + CanaryController + MetricsStore
        CLI (argparse) and bench (synthetic simulation) sit on top of it.
```

| Component | File | Responsibility | Key invariant / rule |
|---|---|---|---|
| Data model | `ml_deploy_ops/models.py` | `ModelVersion`, `EvalResult`, `Deployment`, `RequestMetric`, `Stage` | Stages flow registered → evaluating → staging → canary → production / archived / rejected |
| Registry | `ml_deploy_ops/registry.py` | Register/get/list models, optional JSON persistence | At most one `production` version per model name; duplicates rejected |
| Eval gate | `ml_deploy_ops/evaluation.py` | Threshold checks (`min`/`max`) plus baseline regression check | Missing metric = fail; regression beyond `max_regression` (default 0.02) on the primary metric = fail |
| Canary controller | `ml_deploy_ops/deployment.py` | Deploy/shift/promote/rollback, request routing by traffic draw | Active traffic per model ≤ 100%; promotion retires other deployments; auto-rollback on SLO breach |
| Metrics | `ml_deploy_ops/metrics.py` | Latency percentiles, error rate, windowed summaries, alert rules | Alerts need `min_samples` (default 20) before they can fire |
| Control plane | `ml_deploy_ops/service.py` | Facade wiring the four components together | A model `REJECTED` by the gate cannot be deployed |
| CLI | `ml_deploy_ops/cli.py` | `register`, `list`, `eval`, `deploy`, `promote`, `rollback`, `deployments`, `metrics`, `demo`, `bench` | Registry persists via `--state`; deployments/metrics are in-process (see Limitations) |
| Simulation | `ml_deploy_ops/bench.py` | Synthetic rollout: register → eval → canary → traffic → promote or auto-rollback | Seeded RNG (`seed=7`); faulty canary injected every 5th passing model |

## Run / test / bench

```bash
pip install -e .            # or just run from the repo root, stdlib only
pip install pytest          # dev/test only

# CLI (registry persists in registry.json by default; override with --state)
python -m ml_deploy_ops.cli register sentiment 1.0.0 --framework synthetic
python -m ml_deploy_ops.cli eval sentiment 1.0.0 --metric accuracy=0.93
python -m ml_deploy_ops.cli list
python -m ml_deploy_ops.cli demo        # in-process end-to-end demo
python -m ml_deploy_ops.cli bench       # synthetic simulation as JSON

# Tests
python -m pytest -q

# Benchmark / simulation
python -m ml_deploy_ops.bench
```

## Sanity runs

**Pending — not yet run at publish time (see Status note above).** No output is
quoted here, because none was produced: pasting expected or invented numbers
would be dishonest. After running the commands above, the expected *shape* is:

- `python -m pytest -q` → all tests in `tests/` pass (registry, evaluation,
  deployment, metrics, CLI).
- `python -m ml_deploy_ops.bench` → a summary like
  `models / eval_passed / eval_rejected / promoted / rolled_back /
  requests_simulated / mean_latency_ms / p95_latency_ms / elapsed_s` for
  20 synthetic models × 200 simulated requests each (scaled down, in-process).

## Limitations

- **Not a serving system.** No model is loaded, no socket is opened, no real
  inference happens. Artifact URIs are opaque strings; latencies in the bench
  are drawn from a synthetic log-normal distribution.
- **In-process state (except the registry).** Deployments and metrics live in
  memory. Across separate CLI invocations only the JSON registry survives, so
  `deployments` / `metrics` in a fresh CLI process show empty results — the
  CLI prints a note saying so. `demo` and `bench` show the populated path.
- **Single-process, no concurrency control.** The registry rewrites a whole
  JSON file on each change; concurrent writers can clobber each other. No
  locking, no transactions, no database.
- **Simple statistics.** Percentiles are computed over retained points
  (capped per model); alerts are fixed thresholds on error rate / p95, not
  anomaly detection. No drift detection, no data-quality checks.
- **Simulation scale is small and synthetic** (default 20 models × 200
  requests). Throughput figures from `bench` measure this Python loop, not
  any serving infrastructure, and must not be read as production capacity.
- **No auth, tenancy, or audit log.** Eval history is an in-memory list;
  nothing is signed or access-controlled.
- **Tests unverified at publish time** (see Status). They may contain failures
  that only a real run will reveal.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Ayush Gawai.
