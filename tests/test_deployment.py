import pytest

from ml_deploy_ops.deployment import DeploymentError
from ml_deploy_ops.models import Stage
from ml_deploy_ops.service import ControlPlane


def _passing_cp():
    cp = ControlPlane()
    cp.register_model("m", "1.0.0")
    cp.run_eval("m", "1.0.0", "holdout", {"accuracy": 0.95})
    return cp


def test_canary_then_promote():
    cp = _passing_cp()
    dep = cp.deploy_canary("m", "1.0.0", traffic_percent=10)
    assert cp.registry.get("m", "1.0.0").stage == Stage.CANARY
    cp.canary.promote(dep.id)
    assert dep.traffic_percent == 100 and dep.stage == Stage.PRODUCTION
    assert cp.registry.production("m").version == "1.0.0"


def test_traffic_capacity_enforced():
    cp = _passing_cp()
    cp.register_model("m", "2.0.0"); cp.run_eval("m", "2.0.0", "h", {"accuracy": 0.99})
    cp.deploy_canary("m", "1.0.0", traffic_percent=80)
    with pytest.raises(DeploymentError):
        cp.deploy_canary("m", "2.0.0", traffic_percent=30)


def test_invalid_traffic_rejected():
    cp = _passing_cp()
    with pytest.raises(DeploymentError):
        cp.deploy_canary("m", "1.0.0", traffic_percent=0)
    with pytest.raises(DeploymentError):
        cp.deploy_canary("m", "1.0.0", traffic_percent=101)


def test_route_respects_split():
    cp = _passing_cp()
    dep = cp.deploy_canary("m", "1.0.0", traffic_percent=25)
    assert cp.canary.route("m", 10).id == dep.id
    assert cp.canary.route("m", 50) is None  # remaining traffic is unassigned in this scenario


def test_rollback_archives_canary():
    cp = _passing_cp()
    dep = cp.deploy_canary("m", "1.0.0", traffic_percent=10)
    cp.canary.rollback(dep.id)
    assert not dep.active and dep.traffic_percent == 0
    assert cp.registry.get("m", "1.0.0").stage == Stage.ARCHIVED


def test_auto_rollback_on_error_breach():
    cp = _passing_cp()
    dep = cp.deploy_canary("m", "1.0.0", traffic_percent=10)
    for _ in range(50):
        cp.record_request("m", "1.0.0", latency_ms=50, ok=False)
    assert cp.canary.auto_rollback_check(dep.id, min_samples=20) is True
    assert not dep.active


def test_no_auto_rollback_when_healthy():
    cp = _passing_cp()
    dep = cp.deploy_canary("m", "1.0.0", traffic_percent=10)
    for _ in range(50):
        cp.record_request("m", "1.0.0", latency_ms=50, ok=True)
    assert cp.canary.auto_rollback_check(dep.id, min_samples=20) is False
    assert dep.active
