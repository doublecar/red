@echo off
REM OCR p2 — Windows setup
if not exist ".venv" (
  python -m venv .venv
  .venv\Scripts\pip install --upgrade pip wheel
  .venv\Scripts\pip install -r requirements.txt
) else (
  echo .venv already exists — skipping install.
)
echo Done. Run: run.bat
pause
