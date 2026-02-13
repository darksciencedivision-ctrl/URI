# start_all.ps1
# Starts Ollama (if needed), URI server, and rrr_watcher. Opens UI. Ctrl+C to stop.
# Pure PowerShell.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = "C:\Users\sslaw\URI"
$OllamaHost = "http://localhost:11434/api/tags"
$UriUrl = "http://localhost:8088/"

function Test-OllamaUp {
  try {
    $null = Invoke-RestMethod -Method Get -Uri $OllamaHost -TimeoutSec 3
    return $true
  } catch { return $false }
}

Push-Location $Root

$uriProc = $null
$watchProc = $null
$ollamaProc = $null

try {
  # 1) Start Ollama if not running
  if (-not (Test-OllamaUp)) {
    Write-Host "[START] Ollama not detected. Starting: ollama serve"
    # Start in a separate window so it doesn't get killed accidentally by URI exits
    $ollamaProc = Start-Process -PassThru -WindowStyle Minimized -FilePath "ollama" -ArgumentList "serve"
    Start-Sleep -Milliseconds 800
  } else {
    Write-Host "[START] Ollama already running."
  }

  # 2) Start URI
  Write-Host "[START] Starting URI (python .\uri.py)"
  $py = Get-Command python -ErrorAction SilentlyContinue
  if (-not $py) { throw "Python not found on PATH." }

  # If you rely on venv, URI still works if your stdlib-only script runs. Activate venv is optional here.
  $uriProc = Start-Process -PassThru -WorkingDirectory $Root -NoNewWindow -FilePath "python" -ArgumentList ".\uri.py"

  # 3) Start watcher
  Write-Host "[START] Starting RRR watcher (.\rrr_watcher.ps1)"
  $watchProc = Start-Process -PassThru -WorkingDirectory $Root -NoNewWindow -FilePath "powershell" -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ".\rrr_watcher.ps1"

  # 4) Open browser
  Start-Sleep -Milliseconds 900
  Start-Process $UriUrl

  Write-Host ""
  Write-Host "URI:     $UriUrl"
  Write-Host "Processes:"
  if ($ollamaProc) { Write-Host "  Ollama PID: $($ollamaProc.Id)" }
  if ($uriProc)    { Write-Host "  URI PID:    $($uriProc.Id)" }
  if ($watchProc)  { Write-Host "  Watch PID:  $($watchProc.Id)" }
  Write-Host ""
  Write-Host "Press Ctrl+C to stop everything."

  while ($true) { Start-Sleep -Seconds 1 }

} finally {
  Write-Host "`n[STOP] Shutting down..."

  if ($watchProc -and -not $watchProc.HasExited) {
    try { Stop-Process -Id $watchProc.Id -Force } catch {}
  }
  if ($uriProc -and -not $uriProc.HasExited) {
    try { Stop-Process -Id $uriProc.Id -Force } catch {}
  }

  # Only stop Ollama if we started it in this script
  if ($ollamaProc -and -not $ollamaProc.HasExited) {
    try { Stop-Process -Id $ollamaProc.Id -Force } catch {}
  }

  Pop-Location
  Write-Host "[STOP] Complete."
}
