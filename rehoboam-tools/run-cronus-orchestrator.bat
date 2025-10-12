@echo off
setlocal enabledelayedexpansion

REM Update PROJECT_DIR to match your exact subfolder name containing Scripts and the script subfolders
set PROJECT_DIR=%~dp0
set PYTHON=%PROJECT_DIR%\Scripts\python.exe
set SCRIPT_DIR=%PROJECT_DIR%

echo Starting CRONUS orchestrator.py

pushd "%SCRIPT_DIR%\CRONUS"
%PYTHON% orchestrator.py
popd
echo.
echo orchestrator.py completed.
pause > nul