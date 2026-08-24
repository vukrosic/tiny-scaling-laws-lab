@echo off
setlocal
cd /d "%~dp0"
echo [1/3] Preparing the Python environment...
if not exist ".venv\Scripts\python.exe" (
  where py >nul 2>nul
  if errorlevel 1 (
    python -m venv .venv
  ) else (
    py -3 -m venv .venv
  )
)
if errorlevel 1 exit /b %errorlevel%
".venv\Scripts\python.exe" -u setup_dependencies.py
if errorlevel 1 exit /b %errorlevel%
echo [2/3] Running the CPU scaling experiment (about 4 seconds after setup)...
".venv\Scripts\python.exe" -u scaling_lab.py %*
if errorlevel 1 exit /b %errorlevel%
echo [3/3] Complete.
