"""Latency / error metrics with percentiles and simple alert rules."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from ml_deploy_ops.models import RequestMetric


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return sorted_values[int(rank)]
    return sorted_values[lo] * (hi - rank) + sorted_values[hi] * (rank - lo)


@dataclass
class MetricsSummary:
    model_key: str
    count: int
    error_count: int
    error_rate: float
    mean_latency_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float


@dataclass
class Alert:
    model_key: str
    rule: str
    value: float
    threshold: float
    message: str


class MetricsStore:
    def __init__(self, max_points_per_model: int = 100_000):
        self._data: dict[str, list[RequestMetric]] = {}
        self.max_points = max_points_per_model

    def record(self, metric: RequestMetric) -> None:
        bucket = self._data.setdefault(metric.model_key, [])
        bucket.append(metric)
        if len(bucket) > self.max_points:
            del bucket[: len(bucket) - self.max_points]

    def record_many(self, metrics) -> None:
        for m in metrics:
            self.record(m)

    def summary(self, model_key: str, window_seconds: float | None = None,
                now: float | None = None) -> MetricsSummary:
        now = time.time() if now is None else now
        points = self._data.get(model_key, [])
        if window_seconds is not None:
            points = [p for p in points if now - p.timestamp <= window_seconds]
        if not points:
            return MetricsSummary(model_key, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        lat = sorted(p.latency_ms for p in points)
        errors = sum(1 for p in points if not p.ok)
        return MetricsSummary(
            model_key=model_key, count=len(points), error_count=errors,
            error_rate=errors / len(points),
            mean_latency_ms=sum(lat) / len(lat),
            p50_ms=_percentile(lat, 50), p95_ms=_percentile(lat, 95),
            p99_ms=_percentile(lat, 99), max_ms=lat[-1])

    def check_alerts(self, model_key: str, max_error_rate: float = 0.05,
                     max_p95_ms: float = 500.0, min_samples: int = 20,
                     window_seconds: float | None = None) -> list[Alert]:
        s = self.summary(model_key, window_seconds=window_seconds)
        alerts: list[Alert] = []
        if s.count < min_samples:
            return alerts
        if s.error_rate > max_error_rate:
            alerts.append(Alert(model_key, "error_rate", s.error_rate, max_error_rate,
                                f"error_rate {s.error_rate:.1%} > {max_error_rate:.1%}"))
        if s.p95_ms > max_p95_ms:
            alerts.append(Alert(model_key, "p95_latency", s.p95_ms, max_p95_ms,
                                f"p95 {s.p95_ms:.1f}ms > {max_p95_ms:.1f}ms"))
        return alerts
