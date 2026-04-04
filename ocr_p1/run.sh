#!/usr/bin/env bash
# OCR p1 — run script
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "ERROR: Virtual environment not found. Run 'bash setup.sh' first."
  exit 1
fi

source "$VENV_DIR/bin/activate"

PORT=${PORT:-5100}
echo "==> Starting OCR p1 on http://0.0.0.0:${PORT}"
exec python app.py
