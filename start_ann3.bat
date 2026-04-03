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

:: Check py launcher first (most reliable on Windows)
where py >nul 2>&1
if %errorlevel%==0 (
    for /f "tokens=*" %%v in ('py -3.11 --version 2^>^&1') do set PYVER=%%v
    if "!PYVER:~0,10!"=="Python 3.11" (
        set PYTHON=py -3.11
        echo [OK] Found Python via py launcher: !PYVER!
        goto :found_python
    )
)

:: Check common install paths
for %%P in (
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%ProgramFiles%\Python311\python.exe"
    "%ProgramFiles(x86)%\Python311\python.exe"
    "C:\Python311\python.exe"
    "C:\Python311-32\python.exe"
) do (
    if exist %%P (
        for /f "tokens=*" %%v in ('%%P --version 2^>^&1') do set PYVER=%%v
        if "!PYVER:~0,10!"=="Python 3.11" (
            set PYTHON=%%P
            echo [OK] Found Python at %%P: !PYVER!
            goto :found_python
        )
    )
)

:: Try plain python3 / python
for %%C in (python3 python) do (
    where %%C >nul 2>&1
    if !errorlevel!==0 (
        for /f "tokens=*" %%v in ('%%C --version 2^>^&1') do set PYVER=%%v
        if "!PYVER:~0,10!"=="Python 3.11" (
            set PYTHON=%%C
            echo [OK] Found Python on PATH: !PYVER!
            goto :found_python
        )
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
