@echo off
setlocal EnableDelayedExpansion
rem Calibrate masks from every black* sample in calibration\.
rem   white* partner with the same suffix is passed via --white
rem   names containing "veo" or "flow" -> --name veo, everything else -> --name gemini
set "PY=%~dp0.venv\Scripts\python.exe"
set "TOOL=%~dp0marklift.py"
if not exist "%PY%" call "%~dp0setup.bat" nopause || exit /b 1
set "CAL=%~dp0calibration"
set /a N=0, BAD=0

for %%F in ("%CAL%\black*") do (
    set "FN=%%~nxF"
    set "REST=!FN:~5!"
    set "STEM=%%~nF"
    set "STEMREST=!STEM:~5!"
    set "WHITE="
    if exist "%CAL%\white!REST!" (
        set "WHITE=%CAL%\white!REST!"
    ) else (
        for %%W in ("%CAL%\white!STEMREST!.*") do if /i "%%~nW"=="white!STEMREST!" set "WHITE=%%~fW"
    )
    set "SET=gemini"
    echo !FN! | findstr /i "veo flow" >nul && set "SET=veo"

    echo.
    if defined WHITE (
        echo [!SET!] !FN!  +  white: !WHITE!
        "%PY%" "%TOOL%" calibrate "%%~fF" --white "!WHITE!" --name !SET!
    ) else (
        echo [!SET!] !FN!
        "%PY%" "%TOOL%" calibrate "%%~fF" --name !SET!
    )
    if errorlevel 1 (
        echo   ** calibration FAILED for !FN!
        set /a BAD+=1
    )
    set /a N+=1
)

echo.
if !N!==0 (
    echo No black* samples found in "%CAL%".
) else (
    echo Calibrated !N! sample^(s^), !BAD! failed.
)
echo.
echo Masks now available:
"%PY%" "%TOOL%" masks
echo.
pause
