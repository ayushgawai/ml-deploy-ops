import pytest

from ml_deploy_ops.models import Stage
from ml_deploy_ops.registry import ModelRegistry, RegistryError
from ml_deploy_ops.service import ControlPlane


def test_register_get_and_duplicate():
    cp = ControlPlane()
    m = cp.register_model("clf", "1.0.0", framework="sklearn")
    assert cp.registry.get("clf", "1.0.0").key == "clf:1.0.0"
    assert m.stage == Stage.REGISTERED
    with pytest.raises(RegistryError):
        cp.register_model("clf", "1.0.0")


def test_missing_model_raises():
    with pytest.raises(RegistryError):
        ControlPlane().registry.get("nope", "9")


def test_single_production_invariant():
    r = ModelRegistry()
    from ml_deploy_ops.models import ModelVersion
    r.register(ModelVersion(name="m", version="1"))
    r.register(ModelVersion(name="m", version="2"))
    r.set_stage("m", "1", Stage.PRODUCTION)
    r.set_stage("m", "2", Stage.PRODUCTION)
    assert r.get("m", "1").stage == Stage.ARCHIVED
    assert r.production("m").version == "2"


def test_registry_persistence_roundtrip(tmp_path):
    path = tmp_path / "reg.json"
    cp = ControlPlane(registry_path=path)
    cp.register_model("m", "1.0.0")
    reloaded = ControlPlane(registry_path=path)
    assert len(reloaded.registry) == 1
    assert reloaded.registry.get("m", "1.0.0").key == "m:1.0.0"


def test_latest_and_list_filters():
    cp = ControlPlane()
    cp.register_model("m", "1"); cp.register_model("m", "2"); cp.register_model("x", "1")
    assert cp.registry.latest("m").version == "2"
    assert len(cp.registry.list(name="m")) == 2
