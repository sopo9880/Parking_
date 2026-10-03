@echo off
setlocal
title Parking Research Agent
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  echo Python launcher not found.
  echo Install Python 3.11 or newer and enable Add Python to PATH.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  py -3 -m venv .venv
  if errorlevel 1 goto :fail
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  if errorlevel 1 goto :fail
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 goto :fail
)
".venv\Scripts\python.exe" app.py
if errorlevel 1 goto :fail
exit /b 0
:fail
echo.
echo An error occurred. Copy the last lines and send them to ChatGPT.
pause
exit /b 1
