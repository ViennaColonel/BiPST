# Shared Windows launcher. Paths are relative to this repository, not the shell.
# Configure runtime.local.json once, or activate your Python environment first.
param(
    [Parameter(Mandatory=$true, Position=0)][string]$Script,
    [Parameter(ValueFromRemainingArguments=$true)][string[]]$ScriptArgs
)
$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$LocalConfig = Join-Path $RepoRoot 'runtime.local.json'
$Settings = @{}
if (Test-Path -LiteralPath $LocalConfig) {
    $Settings = Get-Content -LiteralPath $LocalConfig -Raw -Encoding UTF8 | ConvertFrom-Json
}
function Resolve-LocalFile([string]$Value) {
    $Path = [Environment]::ExpandEnvironmentVariables($Value)
    if (-not [IO.Path]::IsPathRooted($Path)) { $Path = Join-Path $RepoRoot $Path }
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "File not found: $Path" }
    return [IO.Path]::GetFullPath($Path)
}
$Wrapper = $null
$Python = $null
if ($env:BIPST_PYTHON) {
    $Python = Resolve-LocalFile $env:BIPST_PYTHON
} elseif ($Settings.python_wrapper) {
    $Wrapper = Resolve-LocalFile $Settings.python_wrapper
} elseif ($Settings.python) {
    $Python = Resolve-LocalFile $Settings.python
} elseif (Test-Path -LiteralPath (Join-Path $RepoRoot '.venv/Scripts/python.exe')) {
    $Python = Join-Path $RepoRoot '.venv/Scripts/python.exe'
} else {
    $Command = Get-Command python -CommandType Application -ErrorAction SilentlyContinue
    if ($Command) { $Python = $Command.Source }
}
if (-not $Python -and -not $Wrapper) {
    throw 'Configure runtime.local.json from runtime.example.json, or activate your Python environment. See README.md.'
}
$Entry = Resolve-LocalFile $Script
Push-Location $RepoRoot
try {
    if ($Wrapper) { & $Wrapper $Entry @ScriptArgs }
    else {
        $env:PYTHONUTF8 = '1'
        $env:PYTHONNOUSERSITE = '1'
        Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
        & $Python $Entry @ScriptArgs
    }
    $Result = $LASTEXITCODE
} finally { Pop-Location }
exit $Result
