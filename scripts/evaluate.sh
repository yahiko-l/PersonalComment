#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/diffupercom_eval.json}"
python -m sdlm.run_summarization "${CONFIG}"
