@REM Copyright (C) 2026 Yusuke Notomi
@REM SPDX-License-Identifier: AGPL-3.0-only

@echo off
cd /d "%~dp0"

call :environment_ready
if defined AMADEUS_ENV_READY goto environment_prepared

call :setup_environment
if errorlevel 1 (
    echo [ERROR] AMADEUS setup or PyTorch/torchvision CUDA verification failed.
    echo [ERROR] See the diagnostic message above.
    call :keep_prompt_open
    exit /b 1
)

:environment_prepared
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] AMADEUS virtual environment was not found.
    call :keep_prompt_open
    exit /b 1
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate the AMADEUS virtual environment.
    call :keep_prompt_open
    exit /b 1
)

set "AMADEUS_SPLASH_TOKEN=%RANDOM%%RANDOM%%RANDOM%"
set "AMADEUS_SPLASH_SECONDS=4"
start "AMADEUS Splash" /min ".venv\Scripts\python.exe" "gui\splash_standalone.py" "%AMADEUS_SPLASH_TOKEN%" "%AMADEUS_SPLASH_SECONDS%"

".venv\Scripts\amadeus.exe"
if errorlevel 1 (
    echo [ERROR] AMADEUS exited with an error.
    call :keep_prompt_open
    exit /b 1
)

echo [AMADEUS] The GUI has closed.
echo [AMADEUS] The AMADEUS virtual environment is active in this prompt.
echo [AMADEUS] Type "amadeus" or "amade" to open the home GUI.
call :keep_prompt_open
exit /b 0

:setup_environment
call :find_uv
if defined UV_EXE goto uv_ready

echo [AMADEUS] Installing uv...
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
call :find_uv

:uv_ready
if not defined UV_EXE (
    echo [ERROR] uv could not be installed or found.
    exit /b 1
)

"%UV_EXE%" run --no-project --python 3.10 python tools\setup_environment.py --uv "%UV_EXE%"
exit /b %errorlevel%

:environment_ready
set "AMADEUS_ENV_READY="
if not exist ".venv\Scripts\python.exe" exit /b 0
if not exist ".venv\Scripts\amadeus.exe" exit /b 0
powershell.exe -NoProfile -Command "$marker='.venv\.amadeus-ready'; $reference=if (Test-Path -LiteralPath $marker) { (Get-Item -LiteralPath $marker).LastWriteTimeUtc } else { (Get-Item -LiteralPath '.venv\Scripts\amadeus.exe').LastWriteTimeUtc }; if ($reference -lt (Get-Item -LiteralPath 'pyproject.toml').LastWriteTimeUtc -or $reference -lt (Get-Item -LiteralPath 'uv.lock').LastWriteTimeUtc -or $reference -lt (Get-Item -LiteralPath 'VERSION').LastWriteTimeUtc) { exit 1 }; if (!(Test-Path -LiteralPath $marker)) { New-Item -ItemType File -Path $marker | Out-Null }" >nul 2>&1
if not errorlevel 1 set "AMADEUS_ENV_READY=1"
exit /b 0

:find_uv
set "UV_EXE="
for /f "delims=" %%I in ('where uv.exe 2^>nul') do if not defined UV_EXE set "UV_EXE=%%I"
if not defined UV_EXE if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV_EXE=%USERPROFILE%\.local\bin\uv.exe"
if not defined UV_EXE if exist "%USERPROFILE%\.cargo\bin\uv.exe" set "UV_EXE=%USERPROFILE%\.cargo\bin\uv.exe"
exit /b 0

:keep_prompt_open
echo [AMADEUS] This command prompt will remain open so you can review or copy messages.
echo [AMADEUS] Type exit and press Enter when you want to close it.
cmd.exe /d /k
exit /b 0
