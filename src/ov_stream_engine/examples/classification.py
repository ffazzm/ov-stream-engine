"""Runnable custom classification task example."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse

from ov_stream_engine.config import load_config
from ov_stream_engine.logic import LabelMatchEvent, PrintEventHandler, ThresholdFilter
from ov_stream_engine.models import (
    CallableModelAdapter,
    DummyClassificationModel,
    ModelRegistry,
    OpenVINOModelAdapter,
)
from ov_stream_engine.pipeline import StreamingEngine
from ov_stream_engine.tasks import ClassificationTask, TaskRegistry


def custom_preprocess(frame: list[int]) -> list[float]:
    return [min(1.0, max(0.0, v / 255.0)) for v in frame]


def custom_postprocess(raw_output: dict[str, float]) -> dict[str, Any]:
    label, score = max(raw_output.items(), key=lambda pair: pair[1])
    return {
        "label": label,
        "score": float(score),
        "scores": {k: float(v) for k, v in raw_output.items()},
    }


def _build_filters(filter_cfgs: list[dict[str, Any]]) -> list[ThresholdFilter]:
    filters: list[ThresholdFilter] = []
    for item in filter_cfgs:
        if item.get("type") == "threshold":
            filters.append(
                ThresholdFilter(
                    score_key=str(item.get("score_key", "score")),
                    min_score=float(item.get("min_score", 0.5)),
                )
            )
    return filters


def _build_events(event_cfgs: list[dict[str, Any]]) -> list[LabelMatchEvent]:
    events: list[LabelMatchEvent] = []
    for item in event_cfgs:
        if item.get("type") == "label_match":
            events.append(
                LabelMatchEvent(
                    target_label=str(item["target_label"]),
                    min_score=float(item.get("min_score", 0.7)),
                )
            )
    return events


def run_from_config(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    classes = cfg.task.params.get("classes", ["bright", "dark"])
    model_registry = ModelRegistry()
    model_registry.register(
        "dummy", lambda **_: DummyClassificationModel(classes=classes)
    )
    model_registry.register(
        "openvino",
        lambda **kwargs: OpenVINOModelAdapter(model_path=kwargs["model_path"]),
    )

    model = model_registry.create(
        cfg.task.model.adapter, model_path=cfg.task.model.model_path
    )

    task_registry = TaskRegistry()
    task_registry.register(
        "classification",
        lambda **kwargs: ClassificationTask(
            name=kwargs["name"],
            task_type=kwargs["task_type"],
            model=kwargs["model"],
            preprocess=kwargs["preprocess"],
            postprocess=kwargs["postprocess"],
        ),
    )

    task = task_registry.create(
        "classification",
        name=cfg.task.name,
        task_type=cfg.task.task_type,
        model=CallableModelAdapter(model.infer),
        preprocess=custom_preprocess,
        postprocess=custom_postprocess,
    )

    engine = StreamingEngine(
        task=task,
        result_filters=_build_filters(cfg.logic.filters),
        event_generators=_build_events(cfg.logic.events),
        event_handlers=[PrintEventHandler()],
    )

    frames = [
        [10, 20, 30, 40],
        [230, 240, 250, 255],
        [100, 110, 120, 130],
    ]

    output = engine.run(frames)
    print(f"processed_results={len(output.results)}")
    for idx, result in enumerate(output.results):
        print(f"result[{idx}] label={result.payload['label']} score={result.payload['score']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ov-stream-engine classification example")
    parser.add_argument(
        "--config",
        default="configs/example_classification.yaml",
        help="Path to engine configuration file",
    )
    args = parser.parse_args()
    run_from_config(args.config)


if __name__ == "__main__":
    main()
