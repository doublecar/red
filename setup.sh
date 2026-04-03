#!/usr/bin/env bash
# ── Ann3 setup script ────────────────────────────────────────────────────
# Creates a Python 3.11 virtual environment and installs all dependencies.
# Run once:  bash setup.sh
# Then start the server:  bash run.sh   (or  source .venv/bin/activate && python app.py)

set -euo pipefail

PYTHON=${PYTHON:-python3.11}
VENV_DIR="${VENV_DIR:-.venv}"

echo "==> Checking Python version …"
PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "    Found: $PY_VERSION"

if [[ "$PY_VERSION" != "3.11"* ]] && [[ "$PY_VERSION" < "3.10" ]]; then
  echo "ERROR: Python 3.11 is required (found $PY_VERSION)."
  echo "       Set PYTHON=/path/to/python3.11 and re-run."
  exit 1
fi

echo "==> Creating virtual environment in $VENV_DIR …"
"$PYTHON" -m venv "$VENV_DIR"

echo "==> Upgrading pip …"
"$VENV_DIR/bin/pip" install --upgrade pip wheel

echo "==> Installing dependencies (this may take a few minutes) …"
"$VENV_DIR/bin/pip" install -r requirements.txt

echo ""
echo "✓ Setup complete."
echo ""
echo "To start Ann3:"
echo "    bash run.sh"
echo ""
echo "Or manually:"
echo "    source $VENV_DIR/bin/activate"
echo "    python app.py"
