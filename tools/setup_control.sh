#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m venv .venv
PIP_CACHE_DIR="$PWD/runtime/cache/pip" .venv/bin/python -m pip install -r requirements-control.txt
.venv/bin/python -m pip check
for check in privacy quality evaluation track_review batch person_library qwen web; do
  .venv/bin/python -m "cli.check_$check"
done
