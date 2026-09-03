#!/usr/bin/env bash
set -euo pipefail

# Supports Ubuntu 22.04+ and similar Debian-based distros.
# Installs Python dependencies and local editable package.

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e .[dev]

echo "Environment ready. Activate with: source .venv/bin/activate"
