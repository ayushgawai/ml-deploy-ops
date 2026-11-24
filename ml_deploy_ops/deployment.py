"""Canary deployments: staged traffic shifting with automatic rollback checks."""
from __future__ import annotations

import time

from ml_deploy_ops.metrics import MetricsStore
from ml_deploy_ops.models import Deployment, Stage
from ml_deploy_ops.registry import ModelRegistry


class DeploymentError(RuntimeError):
    pass


class CanaryController:
    """Manages traffic splits. Invariant: active traffic across a model's
    deployments sums to <= 100; promoting a canary to 100% archives the
    previous production deployment."""

    def __init__(self, registry: ModelRegistry, metrics: MetricsStore | None = None):
        self.registry = registry
        self.metrics = metrics or MetricsStore()
        self._deployments: dict[str, Deployment] = {}  # deployment id -> Deployment
        self._by_model: dict[str, list[str]] = {}  # model name -> deployment ids

    # -- lifecycle ---------------------------------------------------------
    def deploy_canary(self, name: str, version: str, traffic_percent: float = 5.0,
                      replicas: int = 1) -> Deployment:
        if not (0 < traffic_percent <= 100):
            raise DeploymentError("traffic_percent must be in (0, 100]")
        model = self.registry.get(name, version)
        self._check_capacity(name, traffic_percent)
        dep = Deployment(model_key=model.key, stage=Stage.CANARY,
                         traffic_percent=float(traffic_percent), replicas=replicas)
        self._deployments[dep.id] = dep
        self._by_model.setdefault(name, []).append(dep.id)
        self.registry.set_stage(name, version, Stage.CANARY)
        return dep

    def shift_traffic(self, deployment_id: str, traffic_percent: float) -> Deployment:
        dep = self._get(deployment_id)
        if not (0 <= traffic_percent <= 100):
            raise DeploymentError("traffic_percent must be in [0, 100]")
        name = dep.model_key.split(":", 1)[0]
        self._check_capacity(name, traffic_percent, exclude_id=deployment_id)
        dep.traffic_percent = float(traffic_percent)
        dep.updated_at = time.time()
        return dep

    def promote(self, deployment_id: str) -> Deployment:
        dep = self._get(deployment_id)
        name, version = dep.model_key.split(":", 1)
        # Retire every other active deployment for this model.
        for other_id in self._by_model.get(name, []):
            other = self._deployments[other_id]
            if other.id != dep.id and other.active:
                other.active = False
                other.traffic_percent = 0.0
                other.updated_at = time.time()
                o_name, o_version = other.model_key.split(":", 1)
                if self.registry.get(o_name, o_version).stage == Stage.PRODUCTION:
                    self.registry.set_stage(o_name, o_version, Stage.ARCHIVED)
        dep.traffic_percent = 100.0
        dep.stage = Stage.PRODUCTION
        dep.updated_at = time.time()
        self.registry.set_stage(name, version, Stage.PRODUCTION)
        return dep

    def rollback(self, deployment_id: str, reason: str = "") -> Deployment:
        dep = self._get(deployment_id)
        name, version = dep.model_key.split(":", 1)
        dep.active = False
        dep.traffic_percent = 0.0
        dep.updated_at = time.time()
        self.registry.set_stage(name, version, Stage.ARCHIVED)
        # Send all traffic back to the remaining production deployment, if any.
        prod = self.registry.production(name)
        if prod is not None:
            for did in self._by_model.get(name, []):
                d = self._deployments[did]
                if d.model_key == prod.key:
                    d.active, d.traffic_percent = True, 100.0
                    d.updated_at = time.time()
        return dep

    def auto_rollback_check(self, deployment_id: str, max_error_rate: float = 0.05,
                            max_p95_ms: float = 500.0, min_samples: int = 20) -> bool:
        """Return True (and roll back) if the canary breaches its SLOs."""
        dep = self._get(deployment_id)
        alerts = self.metrics.check_alerts(dep.model_key, max_error_rate=max_error_rate,
                                           max_p95_ms=max_p95_ms, min_samples=min_samples)
        if alerts:
            self.rollback(deployment_id, reason="; ".join(a.message for a in alerts))
            return True
        return False

    # -- queries ------------------------------------------------------------
    def list(self, name: str | None = None) -> list[Deployment]:
        deps = list(self._deployments.values())
        if name:
            deps = [d for d in deps if d.model_key.split(":", 1)[0] == name]
        return sorted(deps, key=lambda d: d.created_at)

    def route(self, name: str, draw: float) -> Deployment | None:
        """Pick a deployment for a request given a uniform draw in [0, 100)."""
        if not (0 <= draw < 100):
            raise DeploymentError("draw must be in [0, 100)")
        cumulative = 0.0
        for dep in self.list(name):
            if not dep.active or dep.traffic_percent <= 0:
                continue
            cumulative += dep.traffic_percent
            if draw < cumulative:
                return dep
        return None

    # -- internals ------------------------------------------------------------
    def _get(self, deployment_id: str) -> Deployment:
        try:
            return self._deployments[deployment_id]
        except KeyError:
            raise DeploymentError(f"deployment not found: {deployment_id}") from None

    def _check_capacity(self, name: str, additional: float, exclude_id: str | None = None) -> None:
        total = sum(d.traffic_percent for d in self.list(name)
                    if d.active and d.id != exclude_id)
        if total + additional > 100.0 + 1e-9:
            raise DeploymentError(f"traffic would exceed 100% for {name}: {total:.1f} + {additional:.1f}")
