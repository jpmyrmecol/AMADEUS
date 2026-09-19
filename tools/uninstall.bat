@REM Copyright (C) 2026 Yusuke Notomi
@REM SPDX-License-Identifier: AGPL-3.0-only

@echo off
setlocal
set "AMADEUS_ROOT=%~dp0.."
cd /d "%AMADEUS_ROOT%"

echo [AMADEUS] Removing the local Python environment...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0uninstall_environment.ps1" -ProjectRoot "%AMADEUS_ROOT%"

if errorlevel 1 (
    echo [ERROR] Uninstall failed. Close AMADEUS and try again.
    pause
    exit /b 1
)

echo [AMADEUS] Uninstall complete. Your projects and uv installation were kept.
pause
exit /b 0
