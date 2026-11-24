"""Core data model."""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum


class Stage(str, Enum):
    REGISTERED = "registered"
    EVALUATING = "evaluating"
    STAGING = "staging"
    CANARY = "canary"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    REJECTED = "rejected"


@dataclass
class ModelVersion:
    name: str
    version: str
    artifact_uri: str = ""
    framework: str = "unknown"
    metrics: dict = field(default_factory=dict)
    tags: dict = field(default_factory=dict)
    stage: Stage = Stage.REGISTERED
    created_at: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @property
    def key(self) -> str:
        return f"{self.name}:{self.version}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["stage"] = self.stage.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ModelVersion":
        d = dict(d)
        d["stage"] = Stage(d.get("stage", "registered"))
        return cls(**d)


@dataclass
class EvalResult:
    model_key: str
    dataset: str
    metrics: dict  # e.g. {"accuracy": 0.93, "f1": 0.91}
    passed: bool = False
    reasons: list = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Deployment:
    model_key: str
    stage: Stage = Stage.CANARY
    traffic_percent: float = 0.0  # 0..100 share routed to this deployment
    replicas: int = 1
    active: bool = True
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> dict:
        d = asdict(self)
        d["stage"] = self.stage.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Deployment":
        d = dict(d)
        d["stage"] = Stage(d.get("stage", "canary"))
        return cls(**d)


@dataclass
class RequestMetric:
    model_key: str
    latency_ms: float
    ok: bool = True
    status_code: int = 200
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)
