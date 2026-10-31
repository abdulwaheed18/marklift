@echo off
setlocal
rem Learn the exact Veo/Flow mask from ordinary watermarked videos (no black sample needed).
rem Drag clips/folders onto this file, or run it with no arguments to learn from inbox\.
rem More clips with varied, calm backgrounds = better. Saved into masks\veo\.
set "PY=%~dp0.venv\Scripts\python.exe"
set "TOOL=%~dp0marklift.py"
if not exist "%PY%" call "%~dp0setup.bat" nopause || exit /b 1

if "%~1"=="" (
    "%PY%" "%TOOL%" learn "%~dp0inbox" --name veo
) else (
    "%PY%" "%TOOL%" learn %* --name veo
)
echo.
"%PY%" "%TOOL%" masks
echo.
pause
