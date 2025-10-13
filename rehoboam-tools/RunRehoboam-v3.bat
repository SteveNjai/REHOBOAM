@echo off
setlocal enabledelayedexpansion

REM Update PROJECT_DIR to match your exact subfolder name containing Scripts and the script subfolders
set PROJECT_DIR=%~dp0
set PYTHON=%PROJECT_DIR%\Scripts\python.exe
set SCRIPT_DIR=%PROJECT_DIR%
set SYMBOL_FILE=%PROJECT_DIR%\oracle-spread\symbols.txt

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



