"""Pipeline orchestration for streaming tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable
import logging

from .logic import EventGenerator, EventHandler, FrameCallback, ResultFilter
from .tasks import Task, TaskResult

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RunOutput:
    results: list[TaskResult] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class StreamingEngine:
    """Simple orchestrator decoupling model execution from logic hooks."""

    task: Task
    frame_callbacks: list[FrameCallback] = field(default_factory=list)
    result_filters: list[ResultFilter] = field(default_factory=list)
    event_generators: list[EventGenerator] = field(default_factory=list)
    event_handlers: list[EventHandler] = field(default_factory=list)

    def run(self, frames: Iterable[Any]) -> RunOutput:
        output = RunOutput()

        for frame_index, frame in enumerate(frames):
            for callback in self.frame_callbacks:
                callback(frame_index, frame)

            result = self.task.run(frame)
            if not all(f.keep(result) for f in self.result_filters):
                logger.debug("Frame %s filtered out", frame_index)
                continue

            output.results.append(result)

            for generator in self.event_generators:
                event = generator.generate(frame_index, result)
                if event is None:
                    continue
                output.events.append(event)
                for handler in self.event_handlers:
                    handler(event)

        return output
