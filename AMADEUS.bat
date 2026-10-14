@REM Copyright (C) 2026 Yusuke Notomi
@REM SPDX-License-Identifier: AGPL-3.0-only

@echo off
REM Keep the established v1.1.17 CALL-label flow; Git enforces CRLF on Windows.
set "AMADEUS_LAUNCH_CWD=%CD%"
cd /d "%~dp0"

call :show_version_status
echo [AMADEUS] Location: %CD%

call :environment_ready
if defined AMADEUS_ENV_READY goto environment_prepared

call :setup_environment
if errorlevel 1 (
    echo [ERROR] AMADEUS setup or PyTorch/torchvision accelerator verification failed.
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
set "AMADEUS_GUI_EXIT=%ERRORLEVEL%"
if "%AMADEUS_GUI_EXIT%"=="42" (
    echo [AMADEUS] Update accepted. The updater will restart AMADEUS automatically.
    cd /d "%AMADEUS_LAUNCH_CWD%"
    exit /b 0
)
if "%AMADEUS_GUI_EXIT%"=="43" (
    echo [AMADEUS] Uninstall accepted. A separate Command Prompt will remove AMADEUS.
    cd /d "%AMADEUS_LAUNCH_CWD%"
    exit /b 0
)
if not "%AMADEUS_GUI_EXIT%"=="0" (
    echo [ERROR] AMADEUS exited with an error.
    call :keep_prompt_open
    exit /b %AMADEUS_GUI_EXIT%
)

echo [AMADEUS] GUI closed. Type "amadeus" or "amade" to reopen.
call :keep_prompt_open
exit /b 0

:show_version_status
set "AMADEUS_VERSION_FILE=%~dp0VERSION"
powershell -NoProfile -NonInteractive -Command "$ErrorActionPreference='Stop'; try { $currentText=(Get-Content -Raw -LiteralPath $env:AMADEUS_VERSION_FILE).Trim(); try { $request=[System.Net.HttpWebRequest]::Create('https://github.com/jpmyrmecol/AMADEUS/releases/latest'); $request.Method='HEAD'; $request.AllowAutoRedirect=$true; $request.UserAgent='AMADEUS-Version-Check'; $request.Timeout=2000; $response=$request.GetResponse(); try { $finalUri=$response.ResponseUri } finally { $response.Close() }; $tagPrefix='/jpmyrmecol/AMADEUS/releases/tag/'; $path=$finalUri.AbsolutePath; if ($finalUri.Scheme -ne 'https' -or $finalUri.Host -ne 'github.com' -or -not $path.StartsWith($tagPrefix,[System.StringComparison]::Ordinal)) { throw ('Unexpected latest release URL: ' + $finalUri.AbsoluteUri) }; $latestText=[uri]::UnescapeDataString($path.Substring($tagPrefix.Length)).Trim('/') -replace '^[vV]',''; if ($latestText -notmatch '^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$') { throw ('Invalid latest release tag: ' + $latestText) }; $current=[version]$currentText; $latest=[version]$latestText; if ($latest -gt $current) { Write-Output ('[AMADEUS] Update available: v{0} -> v{1}' -f $currentText,$latestText) } elseif ($latest -lt $current) { Write-Output ('[AMADEUS] Installed version is newer than v{0}' -f $latestText) } } catch { } } catch { }"
set "AMADEUS_VERSION_FILE="
exit /b 0
:setup_environment
setlocal
REM tools\uv_bootstrap.ps1 prints the pinned uv's path and nothing else on
REM stdout; its progress and errors go to stderr and stay visible here.
set "UV_EXE="
for /f "usebackq delims=" %%I in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\uv_bootstrap.ps1" -ProjectRoot "%~dp0."`) do set "UV_EXE=%%I"
if not defined UV_EXE (
    echo [ERROR] uv could not be prepared. See the message above.
    endlocal
    exit /b 1
)

REM Setup may rebuild .venv; use persistent uv-managed Python 3.12
REM outside that environment, as in the v1.1.17 Windows Python policy.
set "VIRTUAL_ENV="
set "SETUP_PYTHON="
for /f "usebackq delims=" %%I in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\windows_python.ps1" -UvExecutable "%UV_EXE%"`) do set "SETUP_PYTHON=%%I"
if not defined SETUP_PYTHON (
    echo [ERROR] uv-managed Python 3.12 could not be prepared for setup.
    endlocal
    exit /b 1
)
echo [AMADEUS] Setup Python (uv-managed): %SETUP_PYTHON%
"%SETUP_PYTHON%" tools\setup_environment.py --uv "%UV_EXE%" --python "%SETUP_PYTHON%"
set "AMADEUS_SETUP_EXIT=%errorlevel%"
endlocal & exit /b %AMADEUS_SETUP_EXIT%

:environment_ready
set "AMADEUS_ENV_READY="
if not exist ".venv\Scripts\python.exe" exit /b 0
if not exist ".venv\Scripts\amadeus.exe" exit /b 0
".venv\Scripts\python.exe" tools\setup_environment.py --check-ready >nul 2>&1
if not errorlevel 1 set "AMADEUS_ENV_READY=1"
exit /b 0

:keep_prompt_open
echo [AMADEUS] This command prompt will remain open so you can review or copy messages.
echo [AMADEUS] Type exit and press Enter when you want to close it.
cmd.exe /d /k
exit /b 0
