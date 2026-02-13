
✅ README.md (copy/paste)

Open:

notepad README.md


Paste:

# URI — Unshackled Research Interface

URI (Unshackled Research Interface) is a local-first AI research environment combining:

- Python-based API + web server (`uri.py`)
- Browser UI (`index.html`)
- File-based RRR pipeline
- PowerShell RRR watcher (`rrr_watcher.ps1`)
- Persistent conversation + entity memory

The system is designed for controlled experimentation with locally hosted Ollama models.

---

## ⚙️ Architecture Overview

**Components**

- **URI Server (Python)**
  - Serves UI
  - Exposes REST API
  - Manages sessions & persistence

- **Browser UI**
  - Chat interface
  - Stats / logs / RRR monitoring

- **RRR Watcher (PowerShell)**
  - Consumes queue files
  - Runs Reasoner → Reducer passes
  - Writes artifacts + logs

- **Ollama**
  - Local model runtime
  - Default host: http://localhost:11434

---

## 📦 Requirements

- Windows PowerShell
- Python 3.x
- Ollama
- Installed models (example):
  - `dolphin-llama3:latest`
  - `deepseek-r1:14b`
  - `qwen2.5:*`

---

## 🚀 Running URI

### Start server

```powershell
cd C:\Users\sslaw\URI
.\.venv\Scripts\python.exe .\uri.py

Start RRR watcher
powershell -ExecutionPolicy Bypass -File .\rrr_watcher.ps1

Open UI

http://localhost:8088/

🧠 Persistence

Runtime data stored locally:

data/conversations.db

data/entity_memory.json

Logs:

logs/system/

logs/dialog/

logs/reducer/

These directories are intentionally excluded from version control.

⚠️ Security Notes

URI is intended for local execution only.

Do NOT expose:

/api/chat

/api/model/switch

/api/rrr/*

to the public internet without authentication.

Bind to 127.0.0.1 unless you understand the implications.

📜 License

This project is released under a dual-license model.

See LICENSE for details.

🧪 Status

Experimental / research-grade software.
No guarantees. No warranties.
Break things responsibly.


Save → close.

---

# ✅ LICENSE (dual license)

Open:

```powershell
notepad LICENSE


Paste your dual license text.

If you want a placeholder structure:

DUAL LICENSE

Option 1:
[Your permissive license text — e.g. MIT]

Option 2:
[Your custom / research / commercial terms]