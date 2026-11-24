"""ControlPlane facade tying registry + gates + canary + metrics together."""
from __future__ import annotations

from pathlib import Path

from ml_deploy_ops.deployment import CanaryController
from ml_deploy_ops.evaluation import EvalGate
from ml_deploy_ops.metrics import MetricsStore
from ml_deploy_ops.models import ModelVersion, RequestMetric, Stage
from ml_deploy_ops.registry import ModelRegistry


class ControlPlane:
    def __init__(self, registry_path: str | Path | None = None, gate: EvalGate | None = None):
        self.registry = ModelRegistry(registry_path)
        self.gate = gate or EvalGate()
        self.metrics = MetricsStore()
        self.canary = CanaryController(self.registry, self.metrics)
        self.eval_history: list = []

    def register_model(self, name: str, version: str, artifact_uri: str = "",
                       framework: str = "unknown", tags: dict | None = None) -> ModelVersion:
        return self.registry.register(ModelVersion(
            name=name, version=version, artifact_uri=artifact_uri,
            framework=framework, tags=tags or {}))

    def run_eval(self, name: str, version: str, dataset: str, metrics: dict):
        model = self.registry.get(name, version)
        self.registry.set_stage(name, version, Stage.EVALUATING)
        baseline = self.registry.production(name)
        verdict = self.gate.evaluate(model.key, dataset, metrics,
                                     baseline_metrics=baseline.metrics if baseline else None)
        self.eval_history.append(verdict.result)
        model.metrics = dict(metrics)
        self.registry.set_stage(name, version, Stage.STAGING if verdict.passed else Stage.REJECTED)
        return verdict

    def deploy_canary(self, name: str, version: str, traffic_percent: float = 5.0):
        model = self.registry.get(name, version)
        if model.stage == Stage.REJECTED:
            raise ValueError(f"{model.key} was rejected by the eval gate; refusing to deploy")
        return self.canary.deploy_canary(name, version, traffic_percent)

    def record_request(self, name: str, version: str, latency_ms: float, ok: bool = True) -> None:
        self.metrics.record(RequestMetric(model_key=f"{name}:{version}", latency_ms=latency_ms, ok=ok))
