"""Pluggable logic hooks for frame/result handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .tasks import TaskResult


class FrameCallback(Protocol):
    def __call__(self, frame_index: int, frame: Any) -> None: ...


class ResultFilter(Protocol):
    def keep(self, result: TaskResult) -> bool: ...


class EventGenerator(Protocol):
    def generate(self, frame_index: int, result: TaskResult) -> dict[str, Any] | None: ...


class EventHandler(Protocol):
    def __call__(self, event: dict[str, Any]) -> None: ...


@dataclass(slots=True)
class ThresholdFilter:
    """Filter out results below a score threshold."""

    score_key: str = "score"
    min_score: float = 0.5

    def keep(self, result: TaskResult) -> bool:
        score = float(result.payload.get(self.score_key, 0.0))
        return score >= self.min_score


@dataclass(slots=True)
class LabelMatchEvent:
    """Generate an event when the target label appears with enough confidence."""

    target_label: str
    min_score: float = 0.7

    def generate(self, frame_index: int, result: TaskResult) -> dict[str, Any] | None:
        label = result.payload.get("label")
        score = float(result.payload.get("score", 0.0))
        if label != self.target_label or score < self.min_score:
            return None
        return {
            "type": "label.match",
            "frame_index": frame_index,
            "task": result.task_name,
            "label": label,
            "score": score,
        }


class PrintEventHandler:
    """Default event handler for quick inspection."""

    def __call__(self, event: dict[str, Any]) -> None:
        print(f"[event] {event}")
