@echo off
setlocal
rem Remove watermarks from dragged-in files/folders, or from inbox\ if none given.
rem Results go to a "cleaned" subfolder next to the inputs; originals are untouched.
set "PY=%~dp0.venv\Scripts\python.exe"
set "TOOL=%~dp0marklift.py"
if not exist "%PY%" call "%~dp0setup.bat" nopause || exit /b 1
set "J=%NUMBER_OF_PROCESSORS%"
if not defined J set "J=4"

if "%~1"=="" (
    echo Cleaning "%~dp0inbox" ...
    "%PY%" "%TOOL%" clean "%~dp0inbox" -j %J%
) else (
    "%PY%" "%TOOL%" clean %* -j %J%
)
echo.
pause
