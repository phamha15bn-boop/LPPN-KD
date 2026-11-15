param(
    [Parameter(Mandatory = $true)][string]$EqtPython,
    [Parameter(Mandatory = $true)][string]$StudentPython,
    [string]$OutputRoot = 'outputs/smoke_test',
    [switch]$AllowTeacherCpu,
    [switch]$ReuseTrainedDemoTeacher
)

$ErrorActionPreference = 'Stop'
$eqt = (Resolve-Path -LiteralPath $EqtPython).Path
$student = (Resolve-Path -LiteralPath $StudentPython).Path
$flags = @()
if ($AllowTeacherCpu) { $flags += '--allow-teacher-cpu' }
if ($ReuseTrainedDemoTeacher) { $flags += '--reuse-trained-demo-teacher' }
$previousBytecodeFlag = $env:PYTHONDONTWRITEBYTECODE
$env:PYTHONDONTWRITEBYTECODE = '1'
Push-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
try {
    & $student -m scripts.run_smoke --eqt-python $eqt --student-python $student --output-root $OutputRoot @flags
    $status = $LASTEXITCODE
} finally {
    Pop-Location
    $env:PYTHONDONTWRITEBYTECODE = $previousBytecodeFlag
}
exit $status
