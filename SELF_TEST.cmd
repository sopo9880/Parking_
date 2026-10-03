@echo off
setlocal
title Parking Research Agent Self Test
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Environment not found. Run START.cmd once first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" self_test.py
pause
