#!/usr/bin/env bash
# OCR p1 — setup script
set -euo pipefail

PYTHON=${PYTHON:-python3.11}
VENV_DIR="${VENV_DIR:-.venv}"

echo "==> Checking for tesseract …"
if ! command -v tesseract &>/dev/null; then
  echo "    Installing tesseract-ocr …"
  sudo apt-get install -y tesseract-ocr tesseract-ocr-eng
fi
tesseract --version | head -1

echo "==> Creating virtual environment in $VENV_DIR …"
"$PYTHON" -m venv "$VENV_DIR"

echo "==> Upgrading pip …"
"$VENV_DIR/bin/pip" install --upgrade pip wheel

echo "==> Installing dependencies …"
"$VENV_DIR/bin/pip" install -r requirements.txt

echo ""
echo "✓ Setup complete. Run:  bash run.sh"
