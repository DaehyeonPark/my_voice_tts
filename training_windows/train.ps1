param(
    [Parameter(Mandatory = $true)] [string]$WslProjectPath,
    [Parameter(Mandatory = $true)] [string]$QwenFinetuningDir
)

$command = "cd '$WslProjectPath' && QWEN_FINETUNING_DIR='$QwenFinetuningDir' bash training_windows/train.sh"
wsl.exe bash -lc $command
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
