@echo off
rem Playable SUMO demo of the EV-yielding force model (double-click me).
rem Optional args pass through, e.g.:  PLAY_SUMO_DEMO.bat --mode bluelight
cd /d "%~dp0"
python experiments\play_sumo.py %*
pause
