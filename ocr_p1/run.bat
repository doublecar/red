@echo off
REM OCR p1 — Windows run script

if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found. Run setup.bat first.
    pause & exit /b 1
)

set PORT=5100
if not "%1"=="" set PORT=%1

echo ==> Starting OCR p1 on http://localhost:%PORT%
echo     Press Ctrl+C to stop.
echo.

call .venv\Scripts\activate.bat
set PORT=%PORT%
python app.py
