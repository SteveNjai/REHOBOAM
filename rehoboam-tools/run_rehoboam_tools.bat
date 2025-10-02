@echo off
setlocal

set PYTHON=%~dp0Scripts\python.exe
set SCRIPT_DIR=%~dp0

:menu
cls
echo REHOBOAM Script Runner
echo =====================
echo 1. Run Zscore_predcitor.py (precit the zscore from history data in history folder)
echo 2. Exit
echo.
set /p choice=Enter your choice (1-2):

if "%choice%"=="1" (
    echo Running Zscore_predictor.py...
    %PYTHON% "%SCRIPT_DIR%zscore_predictor.py"
    echo.
    echo Script completed. Press any key to return to menu.
    pause
    goto menu
) else if "%choice%"=="2" (
    echo Exiting...
    exit /b
) else (
    echo Invalid choice. Please enter 1, or 2.
    echo.
    pause
    goto menu
)
