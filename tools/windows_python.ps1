# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only
# Select a persistent Python 3.10-3.12 for the AMADEUS installer.
# stdout is reserved for the selected absolute Python executable path.
param(
    [Parameter(Mandatory = $true)]
    [string]$UvExecutable
)
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $root '.venv\Scripts\python.exe'

function Resolve-CompatiblePython([string]$Candidate) {
    if ([string]::IsNullOrWhiteSpace($Candidate) -or
        -not (Test-Path -LiteralPath $Candidate -PathType Leaf)) {
        return $null
    }
    try {
        # The Python command must actually execute; a WindowsApps alias is
        # not sufficient. Require an installed interpreter in the supported range.
        $version = & $Candidate -c 'import sys; print("valid" if (3, 10) <= sys.version_info[:2] <= (3, 12) else "invalid")' 2>$null
        if ($LASTEXITCODE -ne 0 -or ([string]$version).Trim() -ne 'valid') {
            return $null
        }

        # If the candidate is from an existing .venv, select the underlying
        # interpreter instead: setup is allowed to delete/rebuild .venv.
        $base = & $Candidate -c 'import sys; print(sys._base_executable)' 2>$null
        if ($LASTEXITCODE -ne 0) { return $null }
        $basePath = ([string]$base).Trim()
        if ([string]::IsNullOrWhiteSpace($basePath) -or
            -not (Test-Path -LiteralPath $basePath -PathType Leaf)) {
            return $null
        }
        $check = & $basePath -c 'import sys; print("valid" if (3, 10) <= sys.version_info[:2] <= (3, 12) else "invalid")' 2>$null
        if ($LASTEXITCODE -ne 0 -or ([string]$check).Trim() -ne 'valid') {
            return $null
        }
        return ([IO.Path]::GetFullPath($basePath))
    } catch {
        return $null
    }
}

# Reuse an existing application interpreter if its base Python is persistent.
$found = Resolve-CompatiblePython $venvPython
if ($found) { Write-Output $found; exit 0 }

# Try installed Windows Python versions without installing anything new.
$launcher = Get-Command py.exe -ErrorAction SilentlyContinue
if ($launcher) {
    foreach ($minor in @(12, 11, 10)) {
        try {
            $result = & $launcher.Source ("-3." + $minor) -c 'import sys; print(sys.executable)' 2>$null
            if ($LASTEXITCODE -ne 0 -or -not $result) { continue }
            $candidate = ([string]$result).Trim()
            $found = Resolve-CompatiblePython $candidate
            if ($found) { Write-Output $found; exit 0 }
        } catch { }
    }
}
$python = Get-Command python.exe -ErrorAction SilentlyContinue
if ($python) {
    $found = Resolve-CompatiblePython $python.Source
    if ($found) { Write-Output $found; exit 0 }
}

# A managed installation is persistent; only uv run creates a temporary
# environment. Look for one first, so a previous successful uv python install
# never has to be repeated.
if (-not (Test-Path -LiteralPath $UvExecutable -PathType Leaf)) {
    [Console]::Error.WriteLine('[ERROR] Pinned uv executable was not found.')
    exit 1
}
function Find-ManagedPython {
    try {
        # --managed-python explicitly includes uv installations; --no-project
        # prevents accidentally selecting AMADEUS .venv.
        $candidateOutput = & $UvExecutable python find --no-project --managed-python 3.12 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $candidateOutput) { return $null }
        $candidate = ([string]$candidateOutput).Trim()
        return (Resolve-CompatiblePython $candidate)
    } catch {
        return $null
    }
}

$found = Find-ManagedPython
if ($found) { Write-Output $found; exit 0 }

try {
    [Console]::Error.WriteLine('[AMADEUS] Installing persistent uv-managed Python 3.12...')
    $installOutput = & $UvExecutable python install 3.12 2>&1
    foreach ($line in $installOutput) {
        [Console]::Error.WriteLine([string]$line)
    }
    if ($LASTEXITCODE -ne 0) {
        [Console]::Error.WriteLine('[ERROR] Installing persistent uv-managed Python 3.12 failed.')
        exit 1
    }
    $found = Find-ManagedPython
    if (-not $found) {
        [Console]::Error.WriteLine('[ERROR] Could not execute the uv-managed Python 3.12 after installation.')
        [Console]::Error.WriteLine('[ERROR] Diagnostic command: uv python find --no-project --managed-python 3.12')
        exit 1
    }
    Write-Output $found
    exit 0
} catch {
    [Console]::Error.WriteLine('[ERROR] Persistent Python provisioning failed: ' + $_.Exception.Message)
    exit 1
}
