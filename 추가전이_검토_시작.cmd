@echo off
set "PYTHON=%~dp0..\venv\Scripts\pythonw.exe"
set "CONFIG=%~dp0..\transition_review_extra_config.json"
if not exist "%PYTHON%" (
  echo Research Python environment is missing.
  pause
  exit /b 1
)
if not exist "%CONFIG%" (
  echo Local extra review configuration is missing.
  pause
  exit /b 1
)
start "" "%PYTHON%" "%~dp0transition_review_ui.py" --config "%CONFIG%"
