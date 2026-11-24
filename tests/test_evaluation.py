from ml_deploy_ops.evaluation import EvalGate
from ml_deploy_ops.models import Stage
from ml_deploy_ops.service import ControlPlane


def test_gate_pass_and_fail_threshold():
    gate = EvalGate(thresholds={"accuracy": ("min", 0.8), "latency_ms": ("max", 100.0)})
    assert gate.evaluate("m:1", "d", {"accuracy": 0.9, "latency_ms": 50}).passed
    v = gate.evaluate("m:1", "d", {"accuracy": 0.7, "latency_ms": 50})
    assert not v.passed and any("accuracy" in r for r in v.reasons)


def test_gate_missing_metric_fails():
    assert not EvalGate().evaluate("m:1", "d", {}).passed


def test_gate_baseline_regression():
    gate = EvalGate(max_regression=0.02)
    assert not gate.evaluate("m:2", "d", {"accuracy": 0.90}, baseline_metrics={"accuracy": 0.95}).passed
    assert gate.evaluate("m:2", "d", {"accuracy": 0.94}, baseline_metrics={"accuracy": 0.95}).passed


def test_control_plane_eval_sets_stage():
    cp = ControlPlane()
    cp.register_model("m", "1")
    assert cp.run_eval("m", "1", "holdout", {"accuracy": 0.95}).passed
    assert cp.registry.get("m", "1").stage == Stage.STAGING
    cp.register_model("m", "2")
    assert not cp.run_eval("m", "2", "holdout", {"accuracy": 0.5}).passed
    assert cp.registry.get("m", "2").stage == Stage.REJECTED


def test_rejected_model_cannot_deploy():
    cp = ControlPlane()
    cp.register_model("m", "1")
    cp.run_eval("m", "1", "holdout", {"accuracy": 0.1})
    try:
        cp.deploy_canary("m", "1")
        assert False, "expected ValueError"
    except ValueError:
        pass
