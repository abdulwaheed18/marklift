@echo off
setlocal
rem One-time setup: creates a private Python environment in .venv and installs
rem the requirements into it. Run it by double-clicking; the other .bat files
rem also call it automatically the first time.
cd /d "%~dp0"

set "PYLAUNCH="
where py >nul 2>nul && set "PYLAUNCH=py -3"
if not defined PYLAUNCH where python >nul 2>nul && set "PYLAUNCH=python"
if not defined PYLAUNCH (
    echo Python 3.10+ was not found.
    echo Install it from https://www.python.org/downloads/ ^(tick "Add to PATH"^)
    echo or run:  winget install Python.Python.3.12
    if /i not "%~1"=="nopause" pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating .venv ...
    %PYLAUNCH% -m venv .venv || goto :fail
)
".venv\Scripts\python.exe" -m pip install --upgrade pip -q || goto :fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt -q || goto :fail
if not exist "inbox" mkdir "inbox"

where ffmpeg >nul 2>nul || (
    echo.
    echo NOTE: ffmpeg was not found. Images work without it, videos need it:
    echo     winget install Gyan.FFmpeg
    echo then open a new window.
)

echo.
echo MarkLift is ready. Drop files in inbox\ and double-click clean.bat.
if /i not "%~1"=="nopause" pause
exit /b 0

:fail
echo Setup failed - see the messages above.
if /i not "%~1"=="nopause" pause
exit /b 1
