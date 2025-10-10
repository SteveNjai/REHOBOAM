@echo off
setlocal enabledelayedexpansion

REM Update PROJECT_DIR to match your exact subfolder name containing Scripts and the script subfolders
set PROJECT_DIR=%~dp0
set PYTHON=%PROJECT_DIR%\Scripts\python.exe
set SCRIPT_DIR=%PROJECT_DIR%
set SYMBOL_FILE=%PROJECT_DIR%\oracle-spread\symbols.txt

:menu
cls
echo REHOBOAM Script Runner
echo =====================
echo 1. Run Zscore_analyzer (show zscore statistics)
echo 2. Run fetch_mt5.py (fetch history for rehoboam_optimizer tool)
echo 3. Run run_optimize.py (optimize on history data for rehoboam_optimizer)
echo 4. Run Oracle-Spread (this tool simulates the PnL for each zscore at that time and find the best zscore for that period)
echo 5. Exit
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
    echo Running Oracle-Spread.py...
    pushd "%SCRIPT_DIR%\oracle-spread"
    for /f "usebackq tokens=1,2 delims=," %%A in ("%SYMBOL_FILE%") do (
        set "SYMBOL1=%%A"
        set "SYMBOL2=%%B"
        set "SYMBOL1=!SYMBOL1:"=!"
        set "SYMBOL2=!SYMBOL2:"=!"
        set "SYMBOL1=!SYMBOL1: =!"
        set "SYMBOL2=!SYMBOL2: =!"

        echo Launching pair: !SYMBOL1! , !SYMBOL2!
        echo ------------------------------------
        start "OracleSpread - !SYMBOL1!-!SYMBOL2!" cmd /c ^
        "cd /d "%SCRIPT_DIR%\oracle-spread" && "%PYTHON%" oracle-spread.py !SYMBOL1! !SYMBOL2!"
    )
    popd
    echo.
    echo All oracle-spread instances launched in parallel windows.
    echo Press any key to return to menu.
    pause >nul
    goto menu


) else if "%choice%"=="5" (
    echo Exiting...
    exit /b
) else (
    echo Invalid choice. Please enter 1, 2, 3, 4 or 5.
    echo.
    pause >nul
    goto menu
)