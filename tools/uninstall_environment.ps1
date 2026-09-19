# Copyright (C) 2026 Yusuke Notomi
# SPDX-License-Identifier: AGPL-3.0-only

param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot
)

$ErrorActionPreference = "Stop"
$cleanProjectRoot = $ProjectRoot.Trim().Trim('"')
$root = [IO.Path]::GetFullPath($cleanProjectRoot)
$venv = [IO.Path]::GetFullPath((Join-Path $root ".venv"))
$expectedVenv = $root.TrimEnd("\") + "\.venv"

if ($venv -ne $expectedVenv) {
    throw "Refusing to remove an unexpected environment path."
}

if (Test-Path -LiteralPath $venv) {
    Get-ChildItem -LiteralPath $venv -File -Recurse -Force -ErrorAction SilentlyContinue |
        ForEach-Object {
            try {
                $_.IsReadOnly = $false
            } catch {
                # Remove-Item -Force will make a final deletion attempt.
            }
        }
    Remove-Item -LiteralPath $venv -Recurse -Force
    Write-Host "[AMADEUS] Removed .venv."
} else {
    Write-Host "[AMADEUS] No .venv was present."
}

$commandDir = [IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA "AMADEUS\bin"))
$commandPath = Join-Path $commandDir "amadeus.cmd"
if (Test-Path -LiteralPath $commandPath) {
    Remove-Item -LiteralPath $commandPath -Force
}
if ((Test-Path -LiteralPath $commandDir) -and
    -not (Get-ChildItem -LiteralPath $commandDir -Force | Select-Object -First 1)) {
    Remove-Item -LiteralPath $commandDir -Force
}

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($null -ne $userPath) {
    $kept = foreach ($entry in ($userPath -split ";")) {
        $trimmed = $entry.Trim()
        if (-not $trimmed) {
            continue
        }
        try {
            $normalized = [IO.Path]::GetFullPath(
                [Environment]::ExpandEnvironmentVariables($trimmed)
            )
        } catch {
            $normalized = $trimmed
        }
        if (-not $normalized.Equals($commandDir, [StringComparison]::OrdinalIgnoreCase)) {
            $trimmed
        }
    }
    [Environment]::SetEnvironmentVariable(
        "Path",
        (($kept -join ";") + $(if ($kept) { ";" } else { "" })),
        "User"
    )
}

if (-not ("Amadeus.EnvironmentBroadcast" -as [type])) {
    Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
namespace Amadeus {
    public static class EnvironmentBroadcast {
        [DllImport("user32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        public static extern IntPtr SendMessageTimeout(
            IntPtr hWnd, uint Msg, UIntPtr wParam, string lParam,
            uint flags, uint timeout, out UIntPtr result);
    }
}
"@
}
$result = [UIntPtr]::Zero
[void][Amadeus.EnvironmentBroadcast]::SendMessageTimeout(
    [IntPtr]0xffff,
    0x001A,
    [UIntPtr]::Zero,
    "Environment",
    0x0002,
    5000,
    [ref]$result
)

Write-Host "[AMADEUS] Removed the global amadeus command."
