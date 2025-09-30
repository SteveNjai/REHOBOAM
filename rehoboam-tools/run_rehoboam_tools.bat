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
echo 4. Run spread_predictor.py (preedict the spread using past history from the price_history.csv)
echo 5. Exit
echo.
set /p choice=Enter your choice (1-5):

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
    echo Running spread_predictor.py...
    %PYTHON% "%SCRIPT_DIR%spread_predictor.py"
    echo.
    echo Script completed. Check spread_plot.png as well as terminal data. Press any key to return to menu.
    pause
    goto menu
) else if "%choice%"=="5" (
    echo Exiting...
    exit /b
) else (
    echo Invalid choice. Please enter 1, 2, 3, 4 or 5.
    echo.
    pause
    goto menu
)
