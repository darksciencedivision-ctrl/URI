# rrr_watcher.ps1
# Watches URI rrr_queue for queries, runs Reason -> Reduce -> optional Reconcile via Ollama, writes URI-formatted responses.
# Pure PowerShell. No modules.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ----------------------------
# CONFIG
# ----------------------------
$Root          = "C:\Users\sslaw\URI"
$QueueDir      = Join-Path $Root "rrr_queue"
$QueueProcessed= Join-Path $QueueDir "processed"
$RespDir       = Join-Path $Root "rrr_responses"
$RespProcessed = Join-Path $RespDir "processed"

$DialogLogDir  = Join-Path $Root "logs\dialog"
$ReducerLogDir = Join-Path $Root "logs\reducer"

$OllamaHost    = "http://localhost:11434"
$OllamaChatApi = "$OllamaHost/api/chat"
$OllamaTagsApi = "$OllamaHost/api/tags"

# Default models (override by editing here)
$ReasonerModel = ""   # if empty: auto-pick first available model
$ReducerModel  = ""   # if empty: auto-pick second available model or same as reasoner

# Watcher loop pacing
$SleepMs       = 300

# Optional: set to $true if you want reconciliation step enabled
$EnableReconcile = $true

# ----------------------------
# HELPERS
# ----------------------------
function Ensure-Dir([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path)) { New-Item -ItemType Directory -Force -Path $Path | Out-Null }
}

function Now-Iso() { (Get-Date).ToString("o") }

function Safe-Json([object]$Obj) {
  try { return ($Obj | ConvertTo-Json -Depth 20 -Compress) } catch { return "{}" }
}

function Write-JsonFile([string]$Dir, [string]$FileName, [object]$Obj) {
  Ensure-Dir $Dir
  $path = Join-Path $Dir $FileName
  ($Obj | ConvertTo-Json -Depth 30) | Set-Content -LiteralPath $path -Encoding UTF8
}

function New-LogName([string]$id, [string]$step) {
  $ts = (Get-Date).ToString("yyyyMMdd_HHmmss_fff")
  return "${ts}_${id}_${step}.json"
}

function Log-Dialog([string]$id, [string]$step, [object]$data) {
  $obj = [ordered]@{
    timestamp = Now-Iso
    id        = $id
    step      = $step
    data      = $data
  }
  Write-JsonFile $DialogLogDir (New-LogName $id $step) $obj
}

function Log-Reducer([string]$id, [string]$step, [object]$data) {
  $obj = [ordered]@{
    timestamp = Now-Iso
    id        = $id
    step      = $step
    data      = $data
  }
  Write-JsonFile $ReducerLogDir (New-LogName $id $step) $obj
}

function Test-OllamaUp {
  try {
    $null = Invoke-RestMethod -Method Get -Uri $OllamaTagsApi -TimeoutSec 3
    return $true
  } catch { return $false }
}

function Get-OllamaModels {
  try {
    $tags = Invoke-RestMethod -Method Get -Uri $OllamaTagsApi -TimeoutSec 5
    $names = @()
    if ($tags.models) {
      foreach ($m in $tags.models) {
        if ($m.name) { $names += [string]$m.name }
      }
    }
    return $names
  } catch {
    return @()
  }
}

function Pick-Models {
  $models = Get-OllamaModels
  if (-not $models -or $models.Count -eq 0) {
    throw "No Ollama models found (api/tags returned none). Pull a model first."
  }

  if ([string]::IsNullOrWhiteSpace($ReasonerModel)) { $script:ReasonerModel = $models[0] }
  if ([string]::IsNullOrWhiteSpace($ReducerModel)) {
    if ($models.Count -ge 2) { $script:ReducerModel = $models[1] } else { $script:ReducerModel = $script:ReasonerModel }
  }
}

function Invoke-OllamaChat([string]$Model, [string]$System, [object[]]$Messages, [hashtable]$Options) {
  $payload = [ordered]@{
    model    = $Model
    stream   = $false
    messages = @()
    options  = $Options
  }

  if (-not [string]::IsNullOrWhiteSpace($System)) {
    $payload.messages += @{ role = "system"; content = $System }
  }
  $payload.messages += $Messages

  $json = $payload | ConvertTo-Json -Depth 20
  return Invoke-RestMethod -Method Post -Uri $OllamaChatApi -ContentType "application/json" -Body $json -TimeoutSec 180
}

function Build-ReasonerPrompt([string]$query, [string]$context, [string]$conversationId) {
  $sys = @"
You are the Reasoner in an RRR pipeline.
Task: Answer the user's query with clear reasoning and evidence-based framing.
Output: A single plain-text answer. No JSON. No meta.
"@
  $user = @"
QUERY:
$query

CONTEXT:
$context

CONVERSATION_ID:
$conversationId
"@
  return @{ system = $sys; messages = @(@{ role="user"; content=$user }) }
}

function Build-ReducerPrompt([string]$query, [string]$reasonerText) {
  $sys = @"
You are the Reducer in an RRR pipeline.
Task: Produce a compact, high-signal final response that preserves key claims and caveats.
Rules:
- Remove fluff.
- Keep structure (bullets ok).
- If the reasoner contains contradictions or uncertainty that must be reconciled, prepend a line:
CONFLICT: <one sentence summary of the conflict>
Then still produce your best reduced answer.
Output: Plain text only.
"@
  $user = @"
ORIGINAL QUERY:
$query

REASONER OUTPUT:
$reasonerText
"@
  return @{ system = $sys; messages = @(@{ role="user"; content=$user }) }
}

function Build-ReconcilePrompt([string]$query, [string]$reasonerText, [string]$reducerText) {
  $sys = @"
You are the Reconciler in an RRR pipeline.
Task: If there is a conflict, resolve it. If not, refine the reducer output.
Output a final plain-text response.
"@
  $user = @"
QUERY:
$query

REASONER OUTPUT:
$reasonerText

REDUCER OUTPUT:
$reducerText

INSTRUCTIONS:
- If the reducer flagged CONFLICT, resolve it and present a single consistent final answer.
- If not, lightly improve the reducer answer without adding new claims.
"@
  return @{ system = $sys; messages = @(@{ role="user"; content=$user }) }
}

function Write-UriResponse([string]$id, [string]$responseText, [string]$source) {
  Ensure-Dir $RespDir
  $out = [ordered]@{
    id        = $id
    response  = $responseText
    source    = $source
    timestamp = Now-Iso
  }
  $path = Join-Path $RespDir "$id.json"
  ($out | ConvertTo-Json -Depth 10) | Set-Content -LiteralPath $path -Encoding UTF8
}

function Move-ToProcessed([string]$path, [string]$processedDir) {
  Ensure-Dir $processedDir
  $name = Split-Path -Leaf $path
  $dest = Join-Path $processedDir $name
  Move-Item -LiteralPath $path -Destination $dest -Force
}

# ----------------------------
# INIT
# ----------------------------
Ensure-Dir $QueueDir
Ensure-Dir $QueueProcessed
Ensure-Dir $RespDir
Ensure-Dir $RespProcessed
Ensure-Dir $DialogLogDir
Ensure-Dir $ReducerLogDir

Write-Host "[RRR_WATCHER] Root: $Root"
Write-Host "[RRR_WATCHER] Queue: $QueueDir"
Write-Host "[RRR_WATCHER] Resp : $RespDir"
Write-Host "[RRR_WATCHER] Ollama: $OllamaHost"

try {
  if (-not (Test-OllamaUp)) {
    Write-Host "[RRR_WATCHER] WARNING: Ollama not reachable right now ($OllamaHost). Watcher will keep running and return error responses when needed."
  } else {
    Pick-Models
    Write-Host "[RRR_WATCHER] Reasoner: $ReasonerModel"
    Write-Host "[RRR_WATCHER] Reducer : $ReducerModel"
  }
} catch {
  Write-Host "[RRR_WATCHER] Model selection warning: $($_.Exception.Message)"
}

# ----------------------------
# MAIN LOOP
# ----------------------------
while ($true) {
  try {
    # If Ollama comes up later, pick models
    if ((-not $ReasonerModel) -or (-not $ReducerModel)) {
      if (Test-OllamaUp) {
        Pick-Models
        Write-Host "[RRR_WATCHER] Models set. Reasoner=$ReasonerModel Reducer=$ReducerModel"
      }
    }

    $files = Get-ChildItem -LiteralPath $QueueDir -Filter "*.json" -File -ErrorAction SilentlyContinue |
             Where-Object { $_.FullName -notlike "*\processed\*" } |
             Sort-Object LastWriteTime

    foreach ($f in $files) {
      $raw = $null
      $q = $null
      $id = $null

      try {
        $raw = Get-Content -LiteralPath $f.FullName -Raw -Encoding UTF8
        $q = $raw | ConvertFrom-Json
        $id = [string]$q.id
        if ([string]::IsNullOrWhiteSpace($id)) { throw "Missing id in query manifest." }

        $queryText = [string]$q.query
        $ctx       = [string]$q.context
        $convId    = [string]$q.conversation_id

        Log-Dialog $id "queue_received" ([ordered]@{
          file = $f.Name
          manifest = $q
        })

        if (-not (Test-OllamaUp)) {
          $msg = "[RRR_WATCHER ERROR] Ollama unreachable at $OllamaHost. Start `ollama serve` and retry."
          Log-Reducer $id "error_ollama_down" @{ message = $msg }
          Write-UriResponse $id $msg "rrr_watcher_error"
          Move-ToProcessed $f.FullName $QueueProcessed
          continue
        }

        if ([string]::IsNullOrWhiteSpace($ReasonerModel) -or [string]::IsNullOrWhiteSpace($ReducerModel)) {
          Pick-Models
        }

        $opts = @{
          temperature = 0.2
          top_p       = 0.9
          num_predict = 1200
          num_ctx     = 8192
        }

        # ---- Reason ----
        $rp = Build-ReasonerPrompt -query $queryText -context $ctx -conversationId $convId
        Log-Dialog $id "reason_request" @{ model=$ReasonerModel; query=$queryText }

        $r1 = Invoke-OllamaChat -Model $ReasonerModel -System $rp.system -Messages $rp.messages -Options $opts
        $reasonText = [string]$r1.message.content
        Log-Dialog $id "reason_response" @{
          model = $ReasonerModel
          content = $reasonText
          raw = $r1
        }

        # ---- Reduce ----
        $dp = Build-ReducerPrompt -query $queryText -reasonerText $reasonText
        Log-Reducer $id "reduce_request" @{ model=$ReducerModel }

        $r2 = Invoke-OllamaChat -Model $ReducerModel -System $dp.system -Messages $dp.messages -Options $opts
        $reducedText = [string]$r2.message.content
        Log-Reducer $id "reduce_response" @{
          model = $ReducerModel
          content = $reducedText
          raw = $r2
        }

        $finalText = $reducedText
        $source = "rrr_cycle"

        # ---- Optional Reconcile ----
        if ($EnableReconcile -and ($reducedText -match "(?im)^\s*CONFLICT\s*:")) {
          Log-Reducer $id "reconcile_triggered" @{ reason="Reducer flagged CONFLICT" }

          $cp = Build-ReconcilePrompt -query $queryText -reasonerText $reasonText -reducerText $reducedText
          $r3 = Invoke-OllamaChat -Model $ReasonerModel -System $cp.system -Messages $cp.messages -Options $opts
          $finalText = [string]$r3.message.content

          Log-Reducer $id "reconcile_response" @{
            model = $ReasonerModel
            content = $finalText
            raw = $r3
          }
        } else {
          Log-Reducer $id "reconcile_skipped" @{ enabled=$EnableReconcile; flaggedConflict=($reducedText -match "(?im)^\s*CONFLICT\s*:") }
        }

        # ---- Write URI response ----
        Write-UriResponse $id $finalText $source
        Log-Dialog $id "response_written" @{ outFile = "$id.json"; bytes = ($finalText.Length) }

        # ---- Move processed query ----
        Move-ToProcessed $f.FullName $QueueProcessed
        Log-Dialog $id "queue_processed" @{ movedTo = "processed\" + $f.Name }

      } catch {
        $err = $_.Exception.Message
        if (-not $id) { $id = "unknown_" + (Get-Date).ToString("yyyyMMdd_HHmmss") }

        Log-Reducer $id "fatal_error" @{ error = $err; file = $f.Name; raw = $raw }
        Write-UriResponse $id ("[RRR_WATCHER ERROR] " + $err) "rrr_watcher_error"

        try { Move-ToProcessed $f.FullName $QueueProcessed } catch {}
      }
    }
  } catch {
    # watcher-level error, keep running
    $msg = $_.Exception.Message
    Write-Host "[RRR_WATCHER] Loop error: $msg"
  }

  Start-Sleep -Milliseconds $SleepMs
}
