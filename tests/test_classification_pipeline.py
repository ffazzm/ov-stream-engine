from ov_stream_engine.logic import LabelMatchEvent, ThresholdFilter
from ov_stream_engine.models import DummyClassificationModel
from ov_stream_engine.pipeline import StreamingEngine
from ov_stream_engine.tasks import ClassificationTask


def test_classification_pipeline_filters_and_generates_events() -> None:
    model = DummyClassificationModel(classes=["bright", "dark"])
    task = ClassificationTask.with_defaults(name="brightness", model=model)

    engine = StreamingEngine(
        task=task,
        result_filters=[ThresholdFilter(min_score=0.81)],
        event_generators=[LabelMatchEvent(target_label="bright", min_score=0.8)],
    )

    output = engine.run(
        [
            [0.1, 0.2, 0.3],
            [0.9, 0.9, 0.8],
            [0.7, 0.6, 0.65],
        ]
    )

    assert len(output.results) == 1
    assert output.results[0].payload["label"] == "bright"
    assert len(output.events) == 1
    assert output.events[0]["type"] == "label.match"
