# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only
# Resolve the pinned uv for AMADEUS.bat, installing it if it is not there yet.
#
# uv resolves uv.lock, so its version is pinned like any other dependency.
# UV_VERSION in the project root is the single source of truth, shared with
# AMADEUS.sh, Colab and tools/setup_environment.py. A uv already installed on
# this machine is used only when it matches; otherwise AMADEUS installs its own
# copy under .uv and leaves the existing installation alone.
#
# Only the resolved path is written to stdout, so the caller can capture it with
# "for /f"; everything else goes to stderr.

param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot
)

$ErrorActionPreference = "Stop"

function Write-Status([string]$Message) {
    [Console]::Error.WriteLine($Message)
}

$root = [IO.Path]::GetFullPath($ProjectRoot.Trim().Trim('"'))
$versionFile = Join-Path $root "UV_VERSION"
if (-not (Test-Path -LiteralPath $versionFile)) {
    Write-Status "[ERROR] UV_VERSION is missing from $root; the AMADEUS folder is incomplete."
    exit 1
}
$required = (Get-Content -LiteralPath $versionFile -Raw).Trim()
if (-not $required) {
    Write-Status "[ERROR] UV_VERSION is empty: $versionFile"
    exit 1
}
$uvDir = Join-Path $root ".uv"

function Get-UvVersion([string]$Path) {
    # "uv 1.2.3 (abc1234 2026-01-01)" -> "1.2.3"
    if (-not $Path -or -not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return ""
    }
    try {
        $reported = & $Path --version 2>$null | Select-Object -First 1
    } catch {
        return ""
    }
    if (-not $reported) {
        return ""
    }
    $fields = ([string]$reported).Trim() -split "\s+"
    if ($fields.Count -lt 2) {
        return ""
    }
    return $fields[1]
}

function Find-PinnedUv {
    $candidates = @(
        (Join-Path $uvDir "uv.exe"),
        (Join-Path $uvDir "bin\uv.exe")
    )
    $onPath = Get-Command "uv.exe" -CommandType Application -ErrorAction SilentlyContinue |
        ForEach-Object { $_.Source }
    if ($onPath) {
        $candidates += $onPath
    }
    if ($env:USERPROFILE) {
        $candidates += (Join-Path $env:USERPROFILE ".local\bin\uv.exe")
        $candidates += (Join-Path $env:USERPROFILE ".cargo\bin\uv.exe")
    }
    foreach ($candidate in $candidates) {
        if ((Get-UvVersion $candidate) -eq $required) {
            return $candidate
        }
    }
    return ""
}

$uv = Find-PinnedUv
if (-not $uv) {
    Write-Status "[AMADEUS] Installing uv $required into $uvDir ..."
    Write-Status "[AMADEUS] Any other uv on this machine is left untouched."
    # INSTALLER_NO_MODIFY_PATH keeps the user's PATH as it is.
    $env:UV_INSTALL_DIR = $uvDir
    $env:INSTALLER_NO_MODIFY_PATH = "1"
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        $installer = Invoke-RestMethod -UseBasicParsing -Uri "https://astral.sh/uv/$required/install.ps1"
        # *>&1 keeps the installer's own chatter off stdout, which carries the
        # resolved path and nothing else.
        $installerOutput = & { Invoke-Expression $installer } *>&1
        foreach ($line in $installerOutput) {
            Write-Status ([string]$line)
        }
    } catch {
        Write-Status "[ERROR] Installing uv $required failed: $($_.Exception.Message)"
        Write-Status "[ERROR] Check your network connection and retry, or install uv $required manually: https://docs.astral.sh/uv/"
        exit 1
    }
    $uv = Find-PinnedUv
}

if (-not $uv) {
    Write-Status "[ERROR] uv $required could not be installed or found."
    Write-Status "[ERROR] Install it manually and re-run: https://docs.astral.sh/uv/"
    exit 1
}

Write-Status "[AMADEUS] Using uv ${required}: $uv"
Write-Output $uv
exit 0
