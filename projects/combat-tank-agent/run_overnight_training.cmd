@echo off
title Combat Tank Offline Overnight Training
wsl.exe -d Ubuntu -- bash -lc "cd \"$(wslpath '%~dp0')\" && bash scripts/overnight_train.sh"
echo.
echo Training process ended. Review the message above before closing this window.
pause
