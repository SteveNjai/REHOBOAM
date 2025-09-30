@echo off
setlocal

set PYTHON=%~dp0Scripts\python.exe
set SCRIPT_DIR=%~dp0

:menu
cls
echo REHOBOAM Script Runner
echo =====================
echo 1. Run create_price_history.py (Generate price_history.csv)
echo 2. Run cointegration_screen.py (Screen cointegrated pairs)
echo 3. Run Optimizator.py (find best parameters from the optimization files)
echo 4. Exit
echo.
set /p choice=Enter your choice (1-4):

if "%choice%"=="1" (
    echo Running create_price_history.py...
    %PYTHON% "%SCRIPT_DIR%create_price_history.py"
    echo.
    echo Script completed. Press any key to return to menu.
    pause
    goto menu
) else if "%choice%"=="2" (
    echo Running cointegration_screen.py...
    %PYTHON% "%SCRIPT_DIR%cointegration_screen.py"
    echo.
    echo Script completed. Check cointegration_results.txt and dendrogram. Press any key to return to menu.
    pause
    goto menu
) else if "%choice%"=="3" (
    echo Running Optimizator.py...
    %PYTHON% "%SCRIPT_DIR%optimizator.py"
    echo.
    echo Script completed. Check aggregated_optimization_results.csv. Press any key to return to menu.
    pause
    goto menu
) else if "%choice%"=="4" (
    echo Exiting...
    exit /b
) else (
    echo Invalid choice. Please enter 1, 2, or 3.
    echo.
    pause
    goto menu
)
