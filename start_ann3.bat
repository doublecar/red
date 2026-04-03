@echo off
setlocal EnableDelayedExpansion

title Ann3 — Patent Figure Annotator

echo ============================================================
echo  Ann3 — Patent Figure Annotator
echo  Powered by Florence-2 + Python 3.11
echo ============================================================
echo.

:: ── Find Python 3.11 ──────────────────────────────────────────────────────
set PYTHON=

:: 1) Check fixed install path (most reliable — works even if not on PATH)
set PYPATH311=%LOCALAPPDATA%\Programs\Python\Python311\python.exe
if exist "%PYPATH311%" (
    set PYTHON="%PYPATH311%"
    echo [OK] Found Python 3.11 at %PYPATH311%
    goto :found_python
)

:: 2) Check other common install paths
for %%P in (
    "%ProgramFiles%\Python311\python.exe"
    "%ProgramFiles(x86)%\Python311\python.exe"
    "C:\Python311\python.exe"
) do (
    if exist %%P (
        set PYTHON=%%P
        echo [OK] Found Python 3.11 at %%P
        goto :found_python
    )
)

:: 3) Try py launcher
where py >nul 2>&1
if %errorlevel%==0 (
    py -3.11 --version >nul 2>&1
    if !errorlevel!==0 (
        set PYTHON=py -3.11
        echo [OK] Found Python 3.11 via py launcher
        goto :found_python
    )
)

:: 4) Check if python on PATH is 3.11
where python >nul 2>&1
if %errorlevel%==0 (
    for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
    if "!PYVER:~0,4!"=="3.11" (
        set PYTHON=python
        echo [OK] Found Python 3.11 on PATH: !PYVER!
        goto :found_python
    )
)

echo [ERROR] Python 3.11 not found.
echo.
echo Pythons currently installed on this PC:
py -0 2>nul || echo   (py launcher not available)
echo.
echo Download Python 3.11.9 (64-bit) directly:
echo   https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
echo.
echo During installation, tick "Add Python 3.11 to PATH" on the first screen.
echo Then close this window and double-click start_ann3.bat again.
echo.
pause
exit /b 1

:found_python

:: ── Setup virtual environment ──────────────────────────────────────────────
if not exist ".venv\Scripts\activate.bat" (
    echo.
    echo [....] Creating virtual environment ...
    %PYTHON% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause & exit /b 1
    )
    echo [OK]   Virtual environment created.
)

:: ── Upgrade pip ────────────────────────────────────────────────────────────
echo.
echo [....] Upgrading pip ...
.venv\Scripts\python.exe -m pip install --upgrade pip wheel --quiet
echo [OK]   pip upgraded.

:: ── Install / verify dependencies ─────────────────────────────────────────
echo.
echo [....] Installing dependencies (first run downloads ~2 GB — please wait) ...
.venv\Scripts\pip.exe install -r requirements.txt --quiet --extra-index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo.
    echo [WARN] Some packages may have failed. Retrying without CPU index ...
    .venv\Scripts\pip.exe install -r requirements.txt --quiet
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed. Check your internet connection.
        pause & exit /b 1
    )
)
echo [OK]   Dependencies installed.

:: ── Note about Florence-2 model download ──────────────────────────────────
echo.
echo NOTE: The Florence-2 model (~1 GB) will be downloaded from HuggingFace
echo       the first time you submit a PDF. This is normal — it is cached
echo       afterwards in %%USERPROFILE%%\.cache\huggingface
echo.

:: ── Launch Flask server ────────────────────────────────────────────────────
echo [....] Starting Ann3 server on http://localhost:5000 ...
echo        Press Ctrl+C in this window to stop.
echo.

:: Open browser after a short delay (runs in background)
start "" cmd /c "ping -n 4 127.0.0.1 >nul && start http://localhost:5000"

:: Start Flask (keeps this window open and shows logs)
.venv\Scripts\python.exe app.py

echo.
echo Server stopped. Press any key to exit.
pause
