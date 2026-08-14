> **Status: historical.** Early-generation research interface, retained for lineage; superseded by later Dark Science Division systems.

# URI — Unshackled Research Interface

URI is a local-first AI research environment for controlled experimentation with locally hosted Ollama models. It combines:

- A Python API + web server (`uri.py`, standard library only)
- A browser chat UI (`index.html`)
- A file-based Reason → Reduce → Reconcile (RRR) pipeline
- A PowerShell RRR watcher (`rrr_watcher.ps1`)
- Persistent conversation storage (SQLite) and entity memory (JSON)

## What it is

The URI server serves the UI, exposes a REST API (`/api/chat`, `/api/stats`, `/api/rrr/*`, `/api/debug`, and more), manages sessions, and persists conversations and entity memory. Model responses can embed `[RRR_QUERY: ...]` tags; the server drops these as queue files that the watcher consumes.

The RRR watcher polls `rrr_queue/` for query files, runs a Reasoner pass, a Reducer pass, and an optional Reconcile pass against the Ollama chat API, then writes URI-formatted responses to `rrr_responses/` and structured JSON logs to `logs/dialog/` and `logs/reducer/`.

## Requirements

- Windows PowerShell
- Python 3.x (`uri.py` uses only the standard library)
- [Ollama](https://ollama.com/) running locally (default host `http://localhost:11434`)
- At least one installed model. Server defaults (configurable via `uri_config.json`):
  - `deepseek-r1:14b` (default model)
  - `mistral:latest`, `llama3.1:latest` (fallbacks)
  - The watcher auto-picks from whatever models are installed unless you set `$ReasonerModel` / `$ReducerModel` in `rrr_watcher.ps1`.

## Quick start

Start everything (Ollama if needed, URI server, watcher, and the browser) with one command:

```powershell
powershell -ExecutionPolicy Bypass -File .\start_all.ps1
```

Or use `run.ps1`, which launches the server and watcher in separate windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

Or start the pieces manually:

```powershell
# 1. Start the server
python .\uri.py

# 2. Start the RRR watcher (separate window)
powershell -ExecutionPolicy Bypass -File .\rrr_watcher.ps1
```

Then open the UI:

```
http://localhost:8088/
```

## How it works

**Components**

- **URI server (Python)** — serves the UI, exposes the REST API, manages sessions and persistence. Loads its system prompt from `system_prompt.txt` and its configuration from `uri_config.json` (created on first run with defaults).
- **Browser UI** — chat interface with stats, logs, and RRR monitoring.
- **RRR watcher (PowerShell)** — consumes queue files from `rrr_queue/`, runs Reasoner → Reducer (→ Reconcile) passes via Ollama, writes responses and logs, and moves handled files to `processed/` subfolders.
- **Ollama** — local model runtime at `http://localhost:11434`.

**Persistence** (local, excluded from version control):

- `data/conversations.db` — SQLite conversation history
- `data/entity_memory.json` — self-observations, beliefs, RRR knowledge
- `logs/system/`, `logs/dialog/`, `logs/reducer/` — server and pipeline logs

## Security notes

URI is intended for local execution only. The default bind address is `0.0.0.0`; set `"host": "127.0.0.1"` in `uri_config.json` unless you understand the implications. Do not expose `/api/chat`, `/api/model/switch`, `/api/config/update`, or `/api/rrr/*` to the public internet without authentication.

## Status

Experimental, research-grade software. No guarantees. No warranties. Break things responsibly.

## License

MIT — see [LICENSE](LICENSE).
