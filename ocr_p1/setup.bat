@echo off
REM OCR p1 — Windows setup script
REM Run once: setup.bat
REM Then: run.bat

echo.
echo === OCR p1 Windows Setup ===
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.11+ from https://python.org
    pause & exit /b 1
)
python --version

REM Check Tesseract
where tesseract >nul 2>&1
if errorlevel 1 (
    if exist "C:\Program Files\Tesseract-OCR\tesseract.exe" (
        echo Found Tesseract at default path.
    ) else (
        echo.
        echo WARNING: tesseract.exe not found on PATH or at default location.
        echo Download and install from:
        echo   https://github.com/UB-Mannheim/tesseract/wiki
        echo Default install path: C:\Program Files\Tesseract-OCR\
        echo.
        echo Press any key to continue setup anyway, or Ctrl+C to abort.
        pause >nul
    )
) else (
    echo Tesseract found: && tesseract --version 2>&1 | findstr tesseract
)

echo.
echo ==> Creating virtual environment...
python -m venv .venv
if errorlevel 1 ( echo ERROR creating venv & pause & exit /b 1 )

echo ==> Upgrading pip...
.venv\Scripts\python -m pip install --upgrade pip wheel

echo ==> Installing dependencies...
.venv\Scripts\pip install -r requirements.txt
if errorlevel 1 ( echo ERROR installing packages & pause & exit /b 1 )

echo.
echo =============================================
echo  Setup complete!  Run:  run.bat
echo =============================================
pause
