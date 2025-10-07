@echo off
setlocal

REM Update PROJECT_DIR to match your exact subfolder name containing Scripts and the script subfolders
set PROJECT_DIR=%~dp0
set PYTHON=%PROJECT_DIR%\Scripts\python.exe
set SCRIPT_DIR=%PROJECT_DIR%

:menu
cls
echo REHOBOAM Script Runner
echo =====================
echo 1. Run Zscore_analyzer (show zscore statistics)
echo 2. Run fetch_mt5.py (fetch history for rehoboam_optimizer tool)
echo 3. Run run_optimize.py (optimize on history data for rehoboam_optimizer)
echo 4. Exit
echo.
set /p choice=Enter your choice (1-4):

if "%choice%"=="1" (
    echo Running zscore_analyzer.py...
    pushd "%SCRIPT_DIR%\zscore_analyzer"
    %PYTHON% zscore_analyzer.py
    popd
    echo.
    echo Script completed. Press any key to return to menu.
    pause >nul
    goto menu
) else if "%choice%"=="2" (
    echo Running fetch_mt5.py...
    pushd "%SCRIPT_DIR%\rehoboam_optimizer"
    %PYTHON% fetch_mt5.py
    popd
    echo.
    echo Script completed. Press any key to return to menu.
    pause >nul
    goto menu
) else if "%choice%"=="3" (
    echo Running run_optimize.py...
    pushd "%SCRIPT_DIR%\rehoboam_optimizer"
    %PYTHON% run_optimize.py
    popd
    echo.
    echo Script completed. Press any key to return to menu.
    pause >nul
    goto menu
) else if "%choice%"=="4" (
    echo Exiting...
    exit /b
) else (
    echo Invalid choice. Please enter 1, 2, 3 or 4.
    echo.
    pause >nul
    goto menu
)