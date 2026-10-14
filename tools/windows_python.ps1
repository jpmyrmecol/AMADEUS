# Return an installed Python 3.10-3.12 without downloading another interpreter.
# stdout contains only the executable path for AMADEUS.bat.
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

# No installed interpreter: uv will download its managed Python 3.12.
Write-Output '3.12'
