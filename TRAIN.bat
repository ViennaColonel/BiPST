@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\runtime.ps1" train.py %*
set "BIPST_EXIT=%ERRORLEVEL%"
if not "%BIPST_NO_PAUSE%"=="1" pause
exit /b %BIPST_EXIT%
