param(
    [switch]$Recreate
)

$ErrorActionPreference = 'Stop'

function Write-Section($msg) {
    Write-Host "`n=== $msg ===" -ForegroundColor Cyan
}

function Get-PythonExe {
    try {
        $py = (Get-Command py -ErrorAction Stop).Path
        return "py"
    } catch {
        try {
            $python = (Get-Command python -ErrorAction Stop).Path
            return "python"
        } catch {
            throw "Python not found. Install Python 3.8+ and ensure it is on PATH or accessible via the 'py' launcher."
        }
    }
}

$venvPath = Join-Path -Path (Get-Location) -ChildPath ".venv1"
$pythonLauncher = Get-PythonExe

Write-Section "Preparing virtual environment at $venvPath"
if (Test-Path $venvPath) {
    if ($Recreate) {
        Write-Host "Removing existing .venv1 ..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $venvPath
    } else {
        Write-Host ".venv1 already exists. Using the existing environment." -ForegroundColor Yellow
    }
}

if (-not (Test-Path $venvPath)) {
    Write-Host "Creating .venv1 ..." -ForegroundColor Green
    & $pythonLauncher -3 -m venv .venv1 2>$null
    if ($LASTEXITCODE -ne 0) {
        # fallback without -3 (in case 'python' was chosen)
        & $pythonLauncher -m venv .venv1
    }
}

$venvPython = Join-Path $venvPath "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Failed to create virtual environment. python.exe not found at $venvPython"
}

Write-Section "Upgrading pip, setuptools, wheel"
& $venvPython -m pip install --upgrade pip setuptools wheel

Write-Section "Installing requirements from requirements.txt"
$reqFile = Join-Path (Get-Location) "requirements.txt"
if (-not (Test-Path $reqFile)) {
    throw "requirements.txt not found at $reqFile"
}

& $venvPython -m pip install -r $reqFile

Write-Section "Done"
Write-Host "To activate this environment in the current session:" -ForegroundColor Green
Write-Host ".\\.venv1\\Scripts\\Activate.ps1" -ForegroundColor Green
