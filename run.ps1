# URI one-command launcher
$ErrorActionPreference = "Stop"

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Here

$port = 8088
$url  = "http://localhost:$port/"

Write-Host "[RUN] Working dir: $Here"

# Start URI server in a new PowerShell window (keeps this window free)
Write-Host "[RUN] Starting URI server..."
Start-Process powershell -ArgumentList @(
  "-NoExit",
  "-Command",
  "Set-Location `"$Here`"; python .\uri.py"
) | Out-Null

Start-Sleep -Seconds 2

# Optional: start local RRR watcher if present
$WatcherCandidates = @(
  Join-Path $Here "rrr_watcher.ps1",
  Join-Path (Split-Path $Here -Parent) "RRR_DUAL_CYCLE\rrr_watcher.ps1",
  Join-Path (Split-Path $Here -Parent) "RRR_DUAL_CYCLE\watcher.ps1"
)

$Watcher = $WatcherCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($Watcher) {
  Write-Host "[RUN] Starting RRR watcher: $Watcher"
  Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location `"$([System.IO.Path]::GetDirectoryName($Watcher))`"; powershell -ExecutionPolicy Bypass -File `"$Watcher`""
  ) | Out-Null
} else {
  Write-Host "[RUN] No watcher found (optional)."
}

# Open browser
Write-Host "[RUN] Opening browser: $url"
Start-Process $url | Out-Null

Write-Host "[RUN] Done."
