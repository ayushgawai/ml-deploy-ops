from ml_deploy_ops.bench import run_simulation
from ml_deploy_ops.metrics import MetricsStore
from ml_deploy_ops.models import RequestMetric


def _fill(store, key="m:1", n=100):
    for i in range(n):
        store.record(RequestMetric(model_key=key, latency_ms=float(i + 1), ok=i < 95))


def test_summary_percentiles_and_errors():
    s = MetricsStore(); _fill(s)
    summ = s.summary("m:1")
    assert summ.count == 100 and summ.error_count == 5
    assert abs(summ.error_rate - 0.05) < 1e-9
    assert 49 < summ.p50_ms < 51
    assert summ.p95_ms > summ.p50_ms and summ.max_ms == 100


def test_empty_summary_is_zeroed():
    summ = MetricsStore().summary("missing:1")
    assert summ.count == 0 and summ.p95_ms == 0.0


def test_window_filter():
    s = MetricsStore()
    s.record(RequestMetric(model_key="m:1", latency_ms=10, timestamp=1000.0))
    s.record(RequestMetric(model_key="m:1", latency_ms=20, timestamp=2000.0))
    assert s.summary("m:1", window_seconds=100, now=2050.0).count == 1


def test_alerts_require_min_samples_and_breach():
    s = MetricsStore()
    for _ in range(10):
        s.record(RequestMetric(model_key="m:1", latency_ms=900, ok=False))
    assert s.check_alerts("m:1", min_samples=20) == []  # not enough samples
    for _ in range(20):
        s.record(RequestMetric(model_key="m:1", latency_ms=900, ok=False))
    rules = {a.rule for a in s.check_alerts("m:1", min_samples=20)}
    assert rules == {"error_rate", "p95_latency"}


def test_bench_simulation_is_deterministic_and_bounded():
    r = run_simulation(models=10, requests_per_model=50, seed=3)
    assert r["models"] == 10
    assert r["eval_passed"] + r["eval_rejected"] == 10
    assert r["promoted"] + r["rolled_back"] == r["eval_passed"]
    assert r["requests_simulated"] == r["eval_passed"] * 50
