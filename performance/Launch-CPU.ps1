param([Parameter(Mandatory=$true)][string]$Model)
$ErrorActionPreference='Stop'
$taskExe=Join-Path $PSScriptRoot 'koboldcpp-nocuda.exe'
if (-not (Test-Path -LiteralPath $taskExe)) { $taskExe=Join-Path $PSScriptRoot '..\candidate\koboldcpp-nocuda.exe' }
& $taskExe --model (Resolve-Path -LiteralPath $Model).Path --usecpu --threads 16 --blasthreads 20 --numa auto --host 127.0.0.1 --port 5001
exit $LASTEXITCODE
