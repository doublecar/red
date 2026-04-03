#!/usr/bin/env bash
# ── Ann3 run script ──────────────────────────────────────────────────────
# Activates the venv created by setup.sh and starts the Flask server.

set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "ERROR: Virtual environment not found. Run 'bash setup.sh' first."
  exit 1
fi

source "$VENV_DIR/bin/activate"

# Optional env vars (override defaults):
#   FLORENCE2_MODEL=microsoft/Florence-2-large-ft
#   FLORENCE2_TASK=<DETAILED_CAPTION>
#   PDF_DPI=150
#   PORT=5000
#   FLASK_DEBUG=1

PORT=${PORT:-5000}

echo "==> Starting Ann3 on http://0.0.0.0:${PORT}"
exec python app.py
