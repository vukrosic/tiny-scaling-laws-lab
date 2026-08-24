#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
printf '%s\n' '[1/3] Preparing the Python environment...'
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
.venv/bin/python -u setup_dependencies.py
printf '%s\n' '[2/3] Running the CPU scaling experiment (about 4 seconds after setup)...'
.venv/bin/python -u scaling_lab.py "$@"
printf '%s\n' '[3/3] Complete.'
