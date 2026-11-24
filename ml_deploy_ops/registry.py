"""Model registry: in-memory store with optional JSON persistence."""
from __future__ import annotations

import json
from pathlib import Path

from ml_deploy_ops.models import ModelVersion, Stage


class RegistryError(KeyError):
    pass


class ModelRegistry:
    def __init__(self, path: str | Path | None = None):
        self._models: dict[str, ModelVersion] = {}
        self.path = Path(path) if path else None
        if self.path and self.path.exists():
            self._load()

    def register(self, model: ModelVersion) -> ModelVersion:
        if model.key in self._models:
            raise RegistryError(f"already registered: {model.key}")
        self._models[model.key] = model
        self._save()
        return model

    def get(self, name: str, version: str) -> ModelVersion:
        try:
            return self._models[f"{name}:{version}"]
        except KeyError:
            raise RegistryError(f"not found: {name}:{version}") from None

    def list(self, name: str | None = None, stage: Stage | None = None) -> list[ModelVersion]:
        out = list(self._models.values())
        if name:
            out = [m for m in out if m.name == name]
        if stage:
            out = [m for m in out if m.stage == stage]
        return sorted(out, key=lambda m: m.created_at)

    def latest(self, name: str) -> ModelVersion | None:
        items = self.list(name=name)
        return items[-1] if items else None

    def production(self, name: str) -> ModelVersion | None:
        items = [m for m in self.list(name=name) if m.stage == Stage.PRODUCTION]
        return items[-1] if items else None

    def set_stage(self, name: str, version: str, stage: Stage) -> ModelVersion:
        m = self.get(name, version)
        # Invariant: at most one production version per model name.
        if stage == Stage.PRODUCTION:
            for other in self.list(name=name):
                if other.key != m.key and other.stage == Stage.PRODUCTION:
                    other.stage = Stage.ARCHIVED
        m.stage = stage
        self._save()
        return m

    def delete(self, name: str, version: str) -> None:
        self.get(name, version)
        del self._models[f"{name}:{version}"]
        self._save()

    def __len__(self) -> int:
        return len(self._models)

    def _save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([m.to_dict() for m in self._models.values()], indent=2), encoding="utf-8")

    def _load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        for d in data:
            m = ModelVersion.from_dict(d)
            self._models[m.key] = m
