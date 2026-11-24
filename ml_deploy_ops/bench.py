"""Simulation / benchmark: synthetic traffic through a canary rollout.

Scaled down honestly: this simulates request routing and latency in-process.
It does not serve a real model, open sockets, or measure production traffic.
"""
from __future__ import annotations

import random
import statistics
import time

from ml_deploy_ops.evaluation import EvalGate
from ml_deploy_ops.service import ControlPlane


def run_simulation(models: int = 20, requests_per_model: int = 200, seed: int = 7) -> dict:
    rng = random.Random(seed)
    cp = ControlPlane(gate=EvalGate(thresholds={"accuracy": ("min", 0.80), "f1": ("min", 0.75)}))
    start = time.perf_counter()
    passed = rejected = promoted = rolled_back = 0
    latencies: list[float] = []

    for i in range(models):
        name, version = f"model-{i}", "1.0.0"
        cp.register_model(name, version, artifact_uri=f"mem://{name}", framework="synthetic")
        accuracy = rng.uniform(0.70, 0.98)
        verdict = cp.run_eval(name, version, "synthetic-holdout",
                              {"accuracy": accuracy, "f1": max(0.0, accuracy - rng.uniform(0, 0.05))})
        if not verdict.passed:
            rejected += 1
            continue
        passed += 1
        dep = cp.deploy_canary(name, version, traffic_percent=10.0)
        # A faulty canary every 5th passing model, to exercise auto-rollback.
        faulty = (i % 5 == 4)
        for _ in range(requests_per_model):
            latency = rng.lognormvariate(3.9, 0.45) * (3.0 if faulty else 1.0)  # ~50ms base
            ok = rng.random() > (0.25 if faulty else 0.01)
            latencies.append(latency)
            cp.record_request(name, version, latency_ms=latency, ok=ok)
        if cp.canary.auto_rollback_check(dep.id):
            rolled_back += 1
        else:
            cp.canary.promote(dep.id)
            promoted += 1

    elapsed = time.perf_counter() - start
    total_requests = len(latencies)
    return {
        "models": models, "eval_passed": passed, "eval_rejected": rejected,
        "promoted": promoted, "rolled_back": rolled_back,
        "requests_simulated": total_requests,
        "mean_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "p95_latency_ms": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 2) if latencies else 0.0,
        "elapsed_s": round(elapsed, 3),
        "requests_per_s_simulated": round(total_requests / elapsed, 1) if elapsed else 0.0,
    }


def main() -> None:
    result = run_simulation()
    print("ml-deploy-ops simulation (synthetic, in-process, scaled down)")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
