@echo off
rem Start-all for the UE5.8 first-person EV demo (double-click me).
rem Brings up: NPC brain -> live 2D view -> bridge socket -> Unreal.
rem Optional args pass through, e.g.:  START_DEMO.bat --game
rem                                    START_DEMO.bat --no-view --seed 8
setlocal
cd /d "%~dp0"

rem --- pick an interpreter that actually has the deps (the WindowsApps
rem --- "python.exe" is a Store stub that exits without running anything).
set "PY="
if defined EMV_PYTHON if exist "%EMV_PYTHON%" set "PY=%EMV_PYTHON%"
if not defined PY if exist "%USERPROFILE%\miniconda3\python.exe" set "PY=%USERPROFILE%\miniconda3\python.exe"
if not defined PY if exist "%USERPROFILE%\anaconda3\python.exe" set "PY=%USERPROFILE%\anaconda3\python.exe"
if not defined PY (
  for /f "delims=" %%p in ('where python 2^>nul') do (
    if not defined PY echo %%p | findstr /i "WindowsApps" >nul || set "PY=%%p"
  )
)
if not defined PY (
  echo [start] no usable Python found. Set EMV_PYTHON to your python.exe, e.g.
  echo         set EMV_PYTHON=C:\Users\%USERNAME%\miniconda3\python.exe
  pause
  exit /b 1
)

"%PY%" -c "import numpy, matplotlib" 2>nul
if errorlevel 1 (
  echo [start] "%PY%" is missing numpy/matplotlib - install with:
  echo         "%PY%" -m pip install -r requirements.txt
  pause
  exit /b 1
)

echo [start] interpreter: %PY%
"%PY%" experiments\start_demo.py %*
pause
