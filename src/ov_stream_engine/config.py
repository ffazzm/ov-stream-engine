"""Configuration loading for ov-stream-engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass(slots=True)
class ModelConfig:
    adapter: str
    model_path: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TaskConfig:
    name: str
    task_type: str
    model: ModelConfig
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LogicConfig:
    filters: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class EngineConfig:
    task: TaskConfig
    logic: LogicConfig = field(default_factory=LogicConfig)
    runtime: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> EngineConfig:
    """Load engine config from YAML or JSON file."""
    cfg_path = Path(path)
    raw = cfg_path.read_text(encoding="utf-8")
    data: dict[str, Any]

    if cfg_path.suffix in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("PyYAML is required for YAML config files") from exc
        data = yaml.safe_load(raw)
    else:
        data = json.loads(raw)

    return load_config_dict(data)


def load_config_dict(data: dict[str, Any]) -> EngineConfig:
    """Load config from a dictionary."""
    task_cfg = data["task"]
    logic_cfg = data.get("logic", {})

    task = TaskConfig(
        name=task_cfg["name"],
        task_type=task_cfg.get("task_type", "generic"),
        model=ModelConfig(
            adapter=task_cfg["model"]["adapter"],
            model_path=task_cfg["model"].get("model_path"),
            params=task_cfg["model"].get("params", {}),
        ),
        params=task_cfg.get("params", {}),
    )

    logic = LogicConfig(
        filters=logic_cfg.get("filters", []),
        events=logic_cfg.get("events", []),
    )

    return EngineConfig(task=task, logic=logic, runtime=data.get("runtime", {}))
