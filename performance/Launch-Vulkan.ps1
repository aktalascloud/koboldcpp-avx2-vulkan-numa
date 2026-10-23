param([Parameter(Mandatory=$true)][string]$Model)
$ErrorActionPreference='Stop'
$taskRoot=$PSScriptRoot
$taskExe=Join-Path $taskRoot 'koboldcpp-nocuda.exe'
if (-not (Test-Path -LiteralPath $taskExe)) { $taskExe=Join-Path $taskRoot '..\candidate\koboldcpp-nocuda.exe' }
$taskModel=(Resolve-Path -LiteralPath $Model).Path
$taskEnv=Get-Content -LiteralPath (Join-Path $taskRoot 'rx580-vulkan-environment.json') -Raw | ConvertFrom-Json
foreach ($taskProperty in $taskEnv.PSObject.Properties) { [Environment]::SetEnvironmentVariable($taskProperty.Name,[string]$taskProperty.Value,'Process') }
& $taskExe --model $taskModel --usevulkan 0 --gpulayers 99 --threads 6 --blasthreads 4 --batchsize 1024 --ubatchsize 1024 --numa auto --host 127.0.0.1 --port 5001
exit $LASTEXITCODE
