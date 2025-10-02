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
echo 5. Solomon Optimzier (uses machine learning to analyze the optimziation files and identify the best parameters)
echo 6. Strategy_auto_tester (automate strategy testing process on mt5)
echo 7. Exit
echo.
set /p choice=Enter your choice (1-7):

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
    echo Running Solomon Optimizer.py...
    %PYTHON% "%SCRIPT_DIR%solomon_optimizer.py"
    echo.
    echo Script completed. Check terminal for best parameters. Press any key to return to menu.
    pause
    goto menu
) else if "%choice%"=="5" (
    echo Running Strategy_auto_tester.py...
    %PYTHON% "%SCRIPT_DIR%strategy_auto_tester.py"
    echo.
    echo Script completed. Check terminal for best parameters. Press any key to return to menu.
    pause
    goto menu
) else if "%choice%"=="7" (
    echo Exiting...
    exit /b
) else (
    echo Invalid choice. Please enter 1, 2, 3, 4, 5, 6 or 7.
    echo.
    pause
    goto menu
)
