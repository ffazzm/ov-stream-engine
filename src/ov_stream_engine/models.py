"""Model adapters and wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol


class ModelAdapter(Protocol):
    """Protocol for model adapters."""

    def infer(self, model_input: Any) -> Any:
        """Run model inference."""


class ModelRegistry:
    """Registry for model adapter builders."""

    def __init__(self) -> None:
        self._builders: dict[str, Callable[..., ModelAdapter]] = {}

    def register(self, name: str, builder: Callable[..., ModelAdapter]) -> None:
        self._builders[name] = builder

    def create(self, name: str, **kwargs: Any) -> ModelAdapter:
        if name not in self._builders:
            raise KeyError(f"Model adapter '{name}' is not registered")
        return self._builders[name](**kwargs)


@dataclass(slots=True)
class CallableModelAdapter:
    """Adapter around a plain callable."""

    fn: Any

    def infer(self, model_input: Any) -> Any:
        return self.fn(model_input)


@dataclass(slots=True)
class OpenVINOModelAdapter:
    """Simple OpenVINO adapter for custom tasks."""

    model_path: str
    device: str = "CPU"

    def __post_init__(self) -> None:
        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"OpenVINO model not found: {self.model_path}")

        try:
            from openvino import Core
        except ImportError as exc:
            raise RuntimeError(
                "openvino package not found. Install OpenVINO Python dependencies first."
            ) from exc

        core = Core()
        model = core.read_model(self.model_path)
        self._compiled_model = core.compile_model(model, self.device)
        self._infer_request = self._compiled_model.create_infer_request()
        self._input = self._compiled_model.inputs[0]
        self._output = self._compiled_model.outputs[0]

    def infer(self, model_input: Any) -> Any:
        self._infer_request.infer({self._input: model_input})
        return self._infer_request.get_tensor(self._output).data


@dataclass(slots=True)
class DummyClassificationModel:
    """Tiny deterministic model for starter examples and tests."""

    classes: list[str]

    def infer(self, model_input: list[float]) -> dict[str, float]:
        base = float(sum(model_input) / max(1, len(model_input)))
        score_a = min(1.0, max(0.0, base))
        score_b = 1.0 - score_a
        if len(self.classes) < 2:
            return {self.classes[0]: 1.0}
        return {self.classes[0]: score_a, self.classes[1]: score_b}
