@echo off
set "PYTHON=%~dp0..\venv\Scripts\pythonw.exe"
set "CONFIG=%~dp0..\human_review_config.json"
if not exist "%PYTHON%" (
  echo Research Python environment is missing.
  pause
  exit /b 1
)
if not exist "%CONFIG%" (
  echo Local review configuration is missing.
  pause
  exit /b 1
)
start "" "%PYTHON%" "%~dp0human_slot_review.py" --config "%CONFIG%"
