# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

# Resolve only uv-managed, persistent Python 3.12 for the Windows launcher.
# This script never searches PATH, py.exe, or the user's Python installations.
# Standard output is reserved for exactly one Python executable path.
param(
    [Parameter(Mandatory = $true)]
    [string]$UvExecutable
)

$ErrorActionPreference = 'Continue'

function Write-Status([string]$Message) {
    [Console]::Error.WriteLine($Message)
}

if (-not (Test-Path -LiteralPath $UvExecutable -PathType Leaf)) {
    Write-Status '[ERROR] The pinned uv executable is missing.'
    exit 1
}

function Find-ManagedPython {
    try {
        # --no-project excludes the application's .venv; --managed-python
        # excludes system/WindowsApps Python installations.
        $output = & $UvExecutable python find --no-project --managed-python 3.12 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $output) {
            return $null
        }
        $candidate = ([string]$output).Trim()
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            return $null
        }
        # Avoid nested quotes in native command arguments: Windows PowerShell
        # 5.1 can otherwise corrupt Python -c expressions.
        & $candidate -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)' 1>$null 2>$null
        if ($LASTEXITCODE -ne 0) {
            return $null
        }
        return ([IO.Path]::GetFullPath($candidate))
    } catch {
        return $null
    }
}

$python = Find-ManagedPython
if (-not $python) {
    Write-Status '[AMADEUS] Installing persistent uv-managed Python 3.12...'
    # Running the uv executable directly does not create a temporary venv.
    & $UvExecutable python install 3.12 1>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Status '[ERROR] Could not install uv-managed Python 3.12.'
        exit 1
    }
    $python = Find-ManagedPython
}

if (-not $python) {
    Write-Status '[ERROR] uv-managed Python 3.12 was not found or could not be executed.'
    Write-Status '[ERROR] Check: uv python find --no-project --managed-python 3.12'
    exit 1
}

Write-Output $python
exit 0
