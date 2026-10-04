@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" transition_refiner_v164.py output
) else (
  python transition_refiner_v164.py output
)
pause
