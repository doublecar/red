@echo off
if not exist ".venv\Scripts\activate.bat" (
  echo Run setup.bat first.
  pause & exit /b 1
)
set PORT=5200
if not "%1"=="" set PORT=%1
echo Starting OCR p2 on http://localhost:%PORT%
call .venv\Scripts\activate.bat
python app.py
