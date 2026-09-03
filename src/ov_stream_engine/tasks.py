"""Task abstractions and built-in task implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .models import ModelAdapter


class Preprocessor(Protocol):
    def __call__(self, frame: Any) -> Any: ...


class Postprocessor(Protocol):
    def __call__(self, raw_output: Any) -> dict[str, Any]: ...


@dataclass(slots=True)
class TaskResult:
    task_name: str
    task_type: str
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


class TaskRegistry:
    """Registry for task builders."""

    def __init__(self) -> None:
        self._builders: dict[str, Callable[..., Task]] = {}

    def register(self, name: str, builder: Callable[..., Task]) -> None:
        self._builders[name] = builder

    def create(self, task_key: str, **kwargs: Any) -> "Task":
        if task_key not in self._builders:
            raise KeyError(f"Task '{task_key}' is not registered")
        return self._builders[task_key](**kwargs)


@dataclass(slots=True)
class Task:
    """Generic task abstraction for custom task types."""

    name: str
    task_type: str
    model: ModelAdapter
    preprocess: Preprocessor
    postprocess: Postprocessor

    def run(self, frame: Any) -> TaskResult:
        model_input = self.preprocess(frame)
        raw_output = self.model.infer(model_input)
        payload = self.postprocess(raw_output)
        return TaskResult(task_name=self.name, task_type=self.task_type, payload=payload)


@dataclass(slots=True)
class ClassificationTask(Task):
    """Convenience classification task."""

    @staticmethod
    def default_preprocess(frame: list[float]) -> list[float]:
        return [float(v) for v in frame]

    @staticmethod
    def default_postprocess(raw_output: dict[str, float]) -> dict[str, Any]:
        if not raw_output:
            return {"label": "unknown", "score": 0.0, "scores": {}}
        label, score = max(raw_output.items(), key=lambda pair: pair[1])
        return {"label": label, "score": float(score), "scores": raw_output}

    @classmethod
    def with_defaults(cls, name: str, model: ModelAdapter) -> "ClassificationTask":
        return cls(
            name=name,
            task_type="classification",
            model=model,
            preprocess=cls.default_preprocess,
            postprocess=cls.default_postprocess,
        )
