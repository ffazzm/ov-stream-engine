# ov-stream-engine

A production-ready starter for building **custom DL Streamer + OpenVINO streaming workflows** in Python.

This project is intentionally task-agnostic. You can use it for **classification, segmentation, detection, pose**, or any custom task type by implementing task preprocess/postprocess and selecting a model adapter.

## Architecture overview

```text
src/ov_stream_engine/
├── config.py            # YAML/JSON config loading
├── logic.py             # Frame callbacks, result filters, event generation/handlers
├── models.py            # Model adapter protocol + OpenVINO and dummy adapters
├── pipeline.py          # Streaming orchestration engine
├── tasks.py             # Generic Task abstraction + classification helper
└── examples/
    └── classification.py  # Runnable non-detection custom task example
```

Core design principles:
- **Task abstraction**: model-specific behavior is isolated in preprocess/postprocess.
- **Registries**: `ModelRegistry` and `TaskRegistry` let you register custom adapters/tasks.
- **Logic decoupling**: filters/events/callbacks live in pluggable logic components.
- **Config-driven execution**: task/model/runtime settings come from config files.

## Prerequisites

- Linux (tested path: Ubuntu 22.04+)
- Python 3.10+
- OpenVINO Runtime (for real OpenVINO model inference)
- Intel DL Streamer (when integrating GStreamer pipelines)

> The included example runs with a dummy model and does not require OpenVINO or DL Streamer.

## Install DL Streamer + OpenVINO (Linux)

### Path A: Intel apt repositories (recommended on Ubuntu)

1. Install Intel OpenVINO repository and runtime packages following Intel docs for your distro.
2. Install GStreamer + DL Streamer packages (package names can vary by release; common names include `intel-dlstreamer` and OpenVINO runtime packages).
3. Verify:

```bash
gst-inspect-1.0 | grep -i -E "gvadetect|gvaclassify|gvainference"
python -c "import openvino; print(openvino.__version__)"
```

### Path B: OpenVINO archive / pip runtime

1. Install OpenVINO runtime from Intel archive installer or Python package.
2. Ensure system GStreamer + DL Streamer plugins are installed separately.
3. Export environment variables if required by your install method.

### Caveats by distro

- Package names differ between Ubuntu versions and Intel release channels.
- Some systems need explicit plugin paths for GStreamer (`GST_PLUGIN_PATH`).
- For GPU/NPU execution, install matching drivers and runtime dependencies.

## Python environment setup

### Quick setup script

```bash
./scripts/setup_env.sh
source .venv/bin/activate
```

### Manual setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e .[dev]
```

Local editable installation is supported with `pip install -e .`.

## Quick start

Run the bundled non-detection task example:

```bash
python -m ov_stream_engine.examples.classification --config configs/example_classification.yaml
```

or

```bash
make run-example
```

or

```bash
python scripts/run_example.py --config configs/example_classification.yaml
```

Expected output pattern:

```text
[event] {'type': 'label.match', 'frame_index': 1, 'task': 'brightness_classifier', ...}
processed_results=2
result[0] label=dark score=...
result[1] label=bright score=...
```

## Configuration format

Example file: `configs/example_classification.yaml`

```yaml
task:
  name: brightness_classifier
  task_type: classification
  model:
    adapter: dummy        # dummy | openvino
    model_path: null      # required when adapter=openvino
  params:
    classes: ["bright", "dark"]
    threshold: 0.60
    event_label: "bright"
    event_threshold: 0.75

logic:
  filters:
    - type: threshold
      score_key: score
      min_score: 0.60
  events:
    - type: label_match
      target_label: bright
      min_score: 0.75
```

## Add a custom model

Register your adapter using `ModelRegistry` (`src/ov_stream_engine/models.py`):

```python
class MyAdapter:
    def infer(self, model_input):
        return my_inference_output

registry = ModelRegistry()
registry.register("my_adapter", lambda **kwargs: MyAdapter())
model = registry.create("my_adapter")
```

Then pass it to your task:

```python
task = ClassificationTask.with_defaults(name="my_task", model=MyAdapter())
```

For OpenVINO IR models (`.xml`/`.bin`), use `OpenVINOModelAdapter(model_path="path/to/model.xml")`.

## Add a custom task

Use the generic `Task` abstraction in `src/ov_stream_engine/tasks.py`, or register builders with `TaskRegistry`:

```python
from ov_stream_engine.tasks import Task, TaskRegistry

task_registry = TaskRegistry()
task_registry.register(
    "segmentation",
    lambda **kwargs: Task(
        name=kwargs["name"],
        task_type="segmentation",
        model=kwargs["model"],
        preprocess=kwargs["preprocess"],
        postprocess=kwargs["postprocess"],
    ),
)
task = task_registry.create(
    "segmentation",
    name="my_segmentation",
    model=my_adapter,
    preprocess=my_preprocess,
    postprocess=my_postprocess,
)
```

This is how you support non-detection workflows while keeping orchestration unchanged.

## Add custom logic plugins

Pluggable logic examples are in `src/ov_stream_engine/logic.py`:

- Frame-level callbacks (`FrameCallback`)
- Result filtering (`ThresholdFilter`)
- Event generation (`LabelMatchEvent`)
- Event handlers (`PrintEventHandler`)

Attach them in `StreamingEngine`:

```python
engine = StreamingEngine(
    task=task,
    frame_callbacks=[my_callback],
    result_filters=[ThresholdFilter(min_score=0.6)],
    event_generators=[LabelMatchEvent(target_label="bright", min_score=0.8)],
    event_handlers=[PrintEventHandler()],
)
```

## Developer ergonomics

- `.env.example` for common local variables.
- `Makefile` targets:
  - `make install`
  - `make dev-install`
  - `make run-example`
  - `make test`
- Starter tests in `tests/` for pipeline and logic behavior.

## Troubleshooting

### `ModuleNotFoundError: ov_stream_engine`
Install package in editable mode:

```bash
python -m pip install -e .
```

### `RuntimeError: openvino package not found`
Install OpenVINO Python runtime and activate the right virtual environment.

### `gst-inspect-1.0` doesn't show DL Streamer plugins
- Verify DL Streamer package installation.
- Check `GST_PLUGIN_PATH` and library dependencies.
- Ensure your shell loaded the environment setup scripts.

### OpenVINO model file not found
Set correct model path in config (`task.model.model_path`) and ensure both `.xml` and `.bin` files are present.

## Development and contribution notes

1. Fork/branch from `main`.
2. Keep changes small and focused.
3. Run checks:

```bash
make test
```

4. Submit PR with:
   - problem statement,
   - approach,
   - run steps,
   - sample output.

## License

MIT (see [LICENSE](LICENSE)).
