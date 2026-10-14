# Return an installed Python 3.10-3.12 without downloading another interpreter.
# stdout contains only the executable path for AMADEUS.bat.
param(
    [Parameter(Mandatory = $true)]
    [string]$UvExecutable
)
$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $root '.venv\Scripts\python.exe'

function Test-Python([string]$path) {
    if (-not $path -or -not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
    try {
        $output = & $path -c 'import sys; print(sys._base_executable if sys.version_info[:2] in [(3,10),(3,11),(3,12)] else "")' 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $output) { return $null }
        $found = ([string]$output).Trim()
        if (Test-Path -LiteralPath $found -PathType Leaf) { return $found }
    } catch { }
    return $null
}

$found = Test-Python $venvPython
if ($found) { Write-Output $found; exit 0 }

$launcher = Get-Command py.exe -ErrorAction SilentlyContinue
if ($launcher) {
    foreach ($minor in @(12, 11, 10)) {
        try {
            $path = & $launcher.Source ("-3." + $minor) -c 'import sys; print(sys.executable)' 2>$null
            if ($LASTEXITCODE -ne 0 -or -not $path) { continue }
            $resolved = ([string]$path).Trim()
            $found = Test-Python $resolved
            if ($found) { Write-Output $found; exit 0 }
        } catch { }
    }
}

$python = Get-Command python.exe -ErrorAction SilentlyContinue
if ($python) {
    $found = Test-Python $python.Source
    if ($found) { Write-Output $found; exit 0 }
}

# Install a persistent uv-managed Python only when no supported installation exists.
# Do not run the environment setup under 'uv run': that creates a temporary venv.
if (-not (Test-Path -LiteralPath $UvExecutable -PathType Leaf)) {
    [Console]::Error.WriteLine('[ERROR] Pinned uv executable was not found.')
    exit 1
}
try {
    $installOutput = & $UvExecutable python install 3.12 2>&1
    foreach ($line in $installOutput) {
        [Console]::Error.WriteLine([string]$line)
    }
    if ($LASTEXITCODE -ne 0) {
        [Console]::Error.WriteLine('[ERROR] Could not install persistent uv-managed Python 3.12.')
        exit 1
    }
    $path = & $UvExecutable python find --system 3.12 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $path) {
        [Console]::Error.WriteLine('[ERROR] Could not find persistent Python 3.12 after installation.')
        exit 1
    }
    $resolved = ([string]$path).Trim()
    $found = Test-Python $resolved
    if (-not $found) {
        [Console]::Error.WriteLine('[ERROR] Installed Python 3.12 cannot be executed.')
        exit 1
    }
    Write-Output $found
    exit 0
} catch {
    [Console]::Error.WriteLine('[ERROR] Persistent Python provisioning failed: ' + $_.Exception.Message)
    exit 1
}
