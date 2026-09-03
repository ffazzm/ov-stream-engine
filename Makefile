.PHONY: install dev-install run-example test

install:
	python -m pip install -r requirements.txt
	python -m pip install -e .

dev-install:
	python -m pip install -r requirements-dev.txt
	python -m pip install -e .[dev]

run-example:
	python -m ov_stream_engine.examples.classification --config configs/example_classification.yaml

test:
	pytest -q
