"""Evaluation gates: compare candidate metrics against thresholds and a baseline."""
from __future__ import annotations

from dataclasses import dataclass, field

from ml_deploy_ops.models import EvalResult

# metric -> (direction, threshold). direction: "min" means value >= threshold.
DEFAULT_THRESHOLDS = {"accuracy": ("min", 0.80)}


@dataclass
class GateVerdict:
    passed: bool
    reasons: list = field(default_factory=list)
    result: EvalResult | None = None


class EvalGate:
    """A gate passes only if every threshold holds and (optionally) the
    candidate is not worse than the baseline by more than `max_regression`
    on the primary metric."""

    def __init__(self, thresholds: dict | None = None, primary_metric: str = "accuracy",
                 max_regression: float = 0.02):
        self.thresholds = dict(thresholds or DEFAULT_THRESHOLDS)
        self.primary_metric = primary_metric
        self.max_regression = float(max_regression)

    def evaluate(self, model_key: str, dataset: str, metrics: dict,
                 baseline_metrics: dict | None = None) -> GateVerdict:
        reasons: list[str] = []
        passed = True
        for metric, (direction, threshold) in self.thresholds.items():
            if metric not in metrics:
                reasons.append(f"missing metric: {metric}")
                passed = False
                continue
            value = float(metrics[metric])
            ok = value >= threshold if direction == "min" else value <= threshold
            reasons.append(f"{metric}={value:.4f} {'≥' if direction == 'min' else '≤'} {threshold} -> {'ok' if ok else 'FAIL'}")
            passed = passed and ok
        if baseline_metrics and self.primary_metric in metrics and self.primary_metric in baseline_metrics:
            delta = float(metrics[self.primary_metric]) - float(baseline_metrics[self.primary_metric])
            ok = delta >= -self.max_regression
            reasons.append(
                f"{self.primary_metric} vs baseline delta={delta:+.4f} "
                f"(allowed regression {-self.max_regression}) -> {'ok' if ok else 'FAIL'}")
            passed = passed and ok
        result = EvalResult(model_key=model_key, dataset=dataset, metrics=dict(metrics),
                            passed=passed, reasons=reasons)
        return GateVerdict(passed=passed, reasons=reasons, result=result)
