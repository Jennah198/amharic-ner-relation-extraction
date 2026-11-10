#!/usr/bin/env bash
# One-command demo launcher for the presentation:
#   FastAPI backend :8000 -> Gradio UI :7860 (-> TensorBoard :6006 with --tb)
# Stop everything with Ctrl+C.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONPATH=src
pids=()
cleanup() { echo; echo "stopping demo..."; kill "${pids[@]}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

uvicorn api.main:app --port 8000 > logs/api.log 2>&1 & pids+=($!)
echo -n "starting API "
until curl -s localhost:8000/health >/dev/null; do echo -n "."; sleep 1; done; echo " ok"

AMNER_API_URL=http://127.0.0.1:8000 python app/gradio_app.py > logs/gradio.log 2>&1 & pids+=($!)
echo -n "starting UI "
until curl -s -o /dev/null localhost:7860; do echo -n "."; sleep 1; done; echo " ok"

if [[ "${1:-}" == "--tb" ]]; then
  tensorboard --logdir runs --port 6006 > logs/tensorboard.log 2>&1 & pids+=($!)
  echo "TensorBoard: http://127.0.0.1:6006"
fi
echo "API docs:    http://127.0.0.1:8000/docs"
echo "Web UI:      http://127.0.0.1:7860"
(xdg-open http://127.0.0.1:7860 >/dev/null 2>&1 || true)
wait
