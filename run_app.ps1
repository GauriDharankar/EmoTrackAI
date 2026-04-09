$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$appPath = Join-Path $projectRoot "app.py"

if (-not (Test-Path $venvPython)) {
    Write-Error "Virtual environment not found at $venvPython"
}

if (-not (Test-Path $appPath)) {
    Write-Error "App file not found at $appPath"
}

& $venvPython -m streamlit run $appPath --server.port 8502
