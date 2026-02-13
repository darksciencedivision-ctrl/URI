#!/usr/bin/env python3
"""
UNSHACKLED RESEARCH INTERFACE (URI)
====================================
Merged build: full research features + robustness improvements.

Research features (original):
  - Full conversation history sent to Ollama
  - System prompt loading (system_prompt.txt)
  - Entity memory: self-observations, beliefs, RRR knowledge
  - Response parsing: [RRR_QUERY:], [SELF_OBSERVE:] tags
  - Context building from accumulated entity state
  - RRR bridge with file-based queue + response watcher
  - Log reader for dialog and reducer logs
  - Persistent SQLite conversation storage

Robustness additions (merged):
  - BASE_DIR anchored paths (no cwd surprises)
  - BOM-tolerant JSON reading (Windows Notepad proof)
  - /api/stats, /api/rrr/latest, /api/rrr/result/<id> endpoints
  - conversation_id returned in chat responses
  - Structured error responses (never crashes the UI)
  - URIServer class holds state (no class-var mutation)
  - "ping" quick health path

Stability hardening (this patch):
  - Uncrashable under browser churn: guards all wfile.write paths
  - Quiet logging (no stack trace spam) to logs/system/uri_server.log
  - Fix session correctness: /api/rrr/query honors payload conversation_id
  - Add /api/debug diagnostics endpoint
  - Track last server error timestamp/message

Author: Dark Science Division
License: Open research — belongs to humanity
"""

from __future__ import annotations

import json
import os
import re
import time
import hashlib
import sqlite3
import threading
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from urllib.parse import urlparse

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

# =============================================================================
# PATH ANCHOR — all relative paths resolve from here
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "uri_config.json"

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    "ollama_host": "http://localhost:11434",
    "model": "deepseek-r1:14b",
    "fallback_models": ["mistral:latest", "llama3.1:latest"],
    "host": "0.0.0.0",
    "port": 8088,
    "rrr_queue_dir": "./rrr_queue",
    "rrr_response_dir": "./rrr_responses",
    "dialog_log_dir": "./logs/dialog",
    "reducer_log_dir": "./logs/reducer",
    "system_log_dir": "./logs/system",
    "conversation_db": "./data/conversations.db",
    "memory_file": "./data/entity_memory.json",
    "system_prompt_file": "./system_prompt.txt",
    "learn_from_conversations": True,
    "memory_enabled": True,
    "rrr_auto_query": True,
    "tts_enabled": True,
    "voice_name": "default",
    "temperature": 0.8,
    "top_p": 0.9,
    "max_tokens": 4096,
    "context_window": 8192,
}


def _resolve(rel: str) -> Path:
    """Resolve a config-relative path against BASE_DIR."""
    return (BASE_DIR / rel).resolve()


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _now_iso() -> str:
    return datetime.now().isoformat()


def _read_json(path: Path) -> Optional[dict]:
    """Read JSON with BOM tolerance (Windows Notepad proof)."""
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _write_json(path: Path, data: Any) -> None:
    _safe_mkdir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_config() -> dict:
    config = DEFAULT_CONFIG.copy()
    user_cfg = _read_json(CONFIG_FILE)
    if isinstance(user_cfg, dict):
        config.update(user_cfg)
    # Backfill any new keys
    changed = False
    for k, v in DEFAULT_CONFIG.items():
        if k not in config:
            config[k] = v
            changed = True
    if changed:
        _write_json(CONFIG_FILE, config)
    return config


def save_config(config: dict) -> None:
    _write_json(CONFIG_FILE, config)


# =============================================================================
# QUIET LOGGING
# =============================================================================

def _log_line(config: dict, line: str) -> None:
    try:
        log_dir = _resolve(config.get("system_log_dir", "./logs/system"))
        _safe_mkdir(log_dir)
        log_path = log_dir / "uri_server.log"
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_path.write_text("", encoding="utf-8", errors="ignore") if not log_path.exists() else None
        with log_path.open("a", encoding="utf-8", errors="ignore") as f:
            f.write(f"[{ts}] {line}\n")
    except Exception:
        # If logging fails, we silently move on. The universe continues.
        pass


def _exc_brief(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"


# =============================================================================
# CONVERSATION MEMORY — Persistent across sessions
# =============================================================================

class ConversationMemory:
    def __init__(self, config: dict):
        self.config = config
        self.db_path = _resolve(config["conversation_db"])
        _safe_mkdir(self.db_path.parent)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        try:
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA synchronous=NORMAL;")
            self.conn.execute("PRAGMA temp_store=MEMORY;")
            self.conn.execute("PRAGMA foreign_keys=ON;")
        except Exception:
            pass
        self.lock = threading.Lock()
        self._init_db()
        self.entity_memory = self._load_entity_memory()

    def _init_db(self) -> None:
        expected = {
            "conversations": {
                "id": ("TEXT", True, True),
                "created_at": ("TEXT", False, False),
                "title": ("TEXT", False, False),
                "message_count": ("INTEGER", False, False),
                "metadata": ("TEXT", False, False),
            },
            "messages": {
                "id": ("INTEGER", True, True),
                "conversation_id": ("TEXT", False, False),
                "role": ("TEXT", False, False),
                "content": ("TEXT", False, False),
                "timestamp": ("TEXT", False, False),
                "metadata": ("TEXT", False, False),
            },
            "rrr_queries": {
                "id": ("INTEGER", True, True),
                "conversation_id": ("TEXT", False, False),
                "query": ("TEXT", False, False),
                "response": ("TEXT", False, False),
                "status": ("TEXT", False, False),
                "created_at": ("TEXT", False, False),
                "resolved_at": ("TEXT", False, False),
                "initiated_by": ("TEXT", False, False),
            },
            "self_observations": {
                "id": ("INTEGER", True, True),
                "observation": ("TEXT", False, False),
                "category": ("TEXT", False, False),
                "timestamp": ("TEXT", False, False),
                "conversation_context": ("TEXT", False, False),
            },
        }

        def _table_exists(name: str) -> bool:
            row = self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (name,),
            ).fetchone()
            return row is not None

        def _get_table_info(name: str) -> List[sqlite3.Row]:
            return self.conn.execute(f"PRAGMA table_info({name})").fetchall()

        def _normalize_type(t: str) -> str:
            tt = (t or "").strip().upper()
            if "INT" in tt:
                return "INTEGER"
            if "CHAR" in tt or "CLOB" in tt or "TEXT" in tt or tt == "":
                return "TEXT"
            if "REAL" in tt or "FLOA" in tt or "DOUB" in tt:
                return "REAL"
            if "BLOB" in tt:
                return "BLOB"
            return tt

        def _schema_ok(table: str) -> (bool, str):
            if not _table_exists(table):
                return False, "missing_table"

            info = _get_table_info(table)
            cols = {r["name"]: r for r in info}

            for col_name, (want_type, want_pk, want_autoinc) in expected[table].items():
                if col_name not in cols:
                    return False, f"missing_column:{table}.{col_name}"

                have_type = _normalize_type(cols[col_name]["type"])
                want_type_norm = _normalize_type(want_type)

                if have_type != want_type_norm:
                    return False, f"type_mismatch:{table}.{col_name} have={have_type} want={want_type_norm}"

                have_pk = int(cols[col_name]["pk"] or 0) == 1
                if want_pk and not have_pk:
                    return False, f"pk_mismatch:{table}.{col_name} missing_pk"
                if (not want_pk) and have_pk:
                    return False, f"pk_mismatch:{table}.{col_name} unexpected_pk"

            if table in ("messages", "rrr_queries", "self_observations"):
                sql_row = self.conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                    (table,),
                ).fetchone()
                sql = (sql_row["sql"] if sql_row and "sql" in sql_row.keys() else "") if sql_row else ""
                if "AUTOINCREMENT" not in (sql or "").upper():
                    return False, f"autoincrement_missing:{table}.id"

            return True, "ok"

        def _backup_tables(tables: List[str]) -> None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for t in tables:
                if _table_exists(t):
                    bak = f"{t}_bak_{stamp}"
                    try:
                        self.conn.execute(f"ALTER TABLE {t} RENAME TO {bak};")
                        print(f"[DB] Backed up table '{t}' -> '{bak}'")
                    except Exception as e:
                        print(f"[DB] Backup rename failed for '{t}': {e}")

        def _recreate_schema() -> None:
            for t in ["messages", "rrr_queries", "self_observations", "conversations"]:
                try:
                    self.conn.execute(f"DROP TABLE IF EXISTS {t};")
                except Exception:
                    pass

            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    title TEXT DEFAULT 'Untitled',
                    message_count INTEGER DEFAULT 0,
                    metadata TEXT DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                );
                CREATE TABLE IF NOT EXISTS rrr_queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT,
                    query TEXT NOT NULL,
                    response TEXT,
                    status TEXT DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    resolved_at TEXT,
                    initiated_by TEXT DEFAULT 'entity'
                );
                CREATE TABLE IF NOT EXISTS self_observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    observation TEXT NOT NULL,
                    category TEXT DEFAULT 'general',
                    timestamp TEXT NOT NULL,
                    conversation_context TEXT
                );
            """)

        with self.lock:
            try:
                self.conn.execute("PRAGMA foreign_keys=OFF;")
            except Exception:
                pass

            problems: List[str] = []
            existing_any = any(_table_exists(t) for t in expected.keys())

            if not existing_any:
                with self.conn:
                    _recreate_schema()
                print("[DB] Initialized fresh schema (no existing tables).")
                try:
                    self.conn.execute("PRAGMA foreign_keys=ON;")
                except Exception:
                    pass
                return

            for t in expected.keys():
                ok, reason = _schema_ok(t)
                if not ok:
                    problems.append(f"{t}:{reason}")

            if problems:
                print("[DB] Schema mismatch detected. Recreating tables (history will be dropped).")
                for p in problems:
                    print(f"     - {p}")
                with self.conn:
                    _backup_tables(["conversations", "messages", "rrr_queries", "self_observations"])
                    _recreate_schema()
                print("[DB] Schema recreated successfully.")
            else:
                with self.conn:
                    self.conn.executescript("""
                        CREATE TABLE IF NOT EXISTS conversations (
                            id TEXT PRIMARY KEY,
                            created_at TEXT NOT NULL,
                            title TEXT DEFAULT 'Untitled',
                            message_count INTEGER DEFAULT 0,
                            metadata TEXT DEFAULT '{}'
                        );
                        CREATE TABLE IF NOT EXISTS messages (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            conversation_id TEXT NOT NULL,
                            role TEXT NOT NULL,
                            content TEXT NOT NULL,
                            timestamp TEXT NOT NULL,
                            metadata TEXT DEFAULT '{}',
                            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                        );
                        CREATE TABLE IF NOT EXISTS rrr_queries (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            conversation_id TEXT,
                            query TEXT NOT NULL,
                            response TEXT,
                            status TEXT DEFAULT 'pending',
                            created_at TEXT NOT NULL,
                            resolved_at TEXT,
                            initiated_by TEXT DEFAULT 'entity'
                        );
                        CREATE TABLE IF NOT EXISTS self_observations (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            observation TEXT NOT NULL,
                            category TEXT DEFAULT 'general',
                            timestamp TEXT NOT NULL,
                            conversation_context TEXT
                        );
                    """)
                print("[DB] Schema OK. No migration needed.")

            try:
                self.conn.execute("PRAGMA foreign_keys=ON;")
            except Exception:
                pass

    def _load_entity_memory(self) -> dict:
        mem_path = _resolve(self.config["memory_file"])
        loaded = _read_json(mem_path)
        if isinstance(loaded, dict):
            return loaded
        return {
            "self_observations": [],
            "learned_preferences": {},
            "interaction_patterns": {},
            "rrr_knowledge": [],
            "formed_beliefs": [],
            "conversation_count": 0,
            "total_messages": 0,
            "created_at": _now_iso(),
        }

    def save_entity_memory(self) -> None:
        _write_json(_resolve(self.config["memory_file"]), self.entity_memory)

    def create_conversation(self, title: Optional[str] = None) -> str:
        with self.lock:
            conv_id = hashlib.sha256(f"{_now_iso()}{time.time()}".encode()).hexdigest()[:12]
            with self.conn:
                self.conn.execute(
                    "INSERT INTO conversations (id, created_at, title) VALUES (?, ?, ?)",
                    (conv_id, _now_iso(), title or f"Session {conv_id[:6]}"),
                )
            self.entity_memory["conversation_count"] = int(self.entity_memory.get("conversation_count", 0)) + 1
            return conv_id

    def add_message(self, conversation_id: str, role: str, content: str, metadata: Optional[dict] = None) -> int:
        with self.lock:
            now = _now_iso()
            meta = json.dumps(metadata or {})
            with self.conn:
                cur = self.conn.execute(
                    "INSERT INTO messages (conversation_id, role, content, timestamp, metadata) VALUES (?, ?, ?, ?, ?)",
                    (conversation_id, role, content, now, meta),
                )
                self.conn.execute(
                    "UPDATE conversations SET message_count = message_count + 1 WHERE id = ?",
                    (conversation_id,),
                )
            self.entity_memory["total_messages"] = int(self.entity_memory.get("total_messages", 0)) + 1
            return int(cur.lastrowid)

    def get_conversation_history(self, conversation_id: str, limit: int = 50) -> List[dict]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT role, content, timestamp, metadata FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
                (conversation_id, limit),
            ).fetchall()
            return [dict(r) for r in reversed(rows)]

    def get_all_conversations(self) -> List[dict]:
        with self.lock:
            rows = self.conn.execute("SELECT * FROM conversations ORDER BY created_at DESC").fetchall()
            return [dict(r) for r in rows]

    def stats(self) -> dict:
        with self.lock:
            convs = self.conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            msgs = self.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        return {"conversations": convs, "messages": msgs}

    def add_self_observation(self, observation: str, category: str = "general", context: Optional[str] = None) -> None:
        observation = (observation or "").strip()
        if not observation:
            return
        with self.lock:
            now = _now_iso()
            with self.conn:
                self.conn.execute(
                    "INSERT INTO self_observations (observation, category, timestamp, conversation_context) VALUES (?, ?, ?, ?)",
                    (observation, category, now, context),
                )
            self.entity_memory.setdefault("self_observations", []).append(
                {"observation": observation, "category": category, "timestamp": now}
            )
            if len(self.entity_memory["self_observations"]) > 500:
                self.entity_memory["self_observations"] = self.entity_memory["self_observations"][-500:]

    def add_rrr_query(self, conversation_id: Optional[str], query: str, initiated_by: str = "entity") -> int:
        query = (query or "").strip()
        if not query:
            return -1
        with self.lock:
            now = _now_iso()
            with self.conn:
                cur = self.conn.execute(
                    "INSERT INTO rrr_queries (conversation_id, query, status, created_at, initiated_by) VALUES (?, ?, 'pending', ?, ?)",
                    (conversation_id, query, now, initiated_by),
                )
            return int(cur.lastrowid)

    def resolve_rrr_query(self, query_id: int, response: str) -> None:
        with self.lock:
            with self.conn:
                self.conn.execute(
                    "UPDATE rrr_queries SET response = ?, status = 'resolved', resolved_at = ? WHERE id = ?",
                    (response, _now_iso(), query_id),
                )


# =============================================================================
# RRR CYCLE INTEGRATION — File-based queue matching NEO-LAB patterns
# =============================================================================

class RRRBridge:
    def __init__(self, config: dict, memory: ConversationMemory):
        self.config = config
        self.memory = memory
        self.queue_dir = _resolve(config["rrr_queue_dir"])
        self.response_dir = _resolve(config["rrr_response_dir"])
        _safe_mkdir(self.queue_dir)
        _safe_mkdir(self.response_dir)
        self._pending_callbacks: Dict[str, Callable[[dict], None]] = {}
        self._pending_lock = threading.Lock()
        self._watcher_thread = threading.Thread(target=self._watch_responses, daemon=True)
        self._watcher_thread.start()

    def submit_query(
        self,
        query: str,
        context: str = "",
        conversation_id: Optional[str] = None,
        callback: Optional[Callable[[dict], None]] = None,
    ) -> str:
        query = (query or "").strip()
        query_id = hashlib.sha256(f"{query}{time.time()}".encode()).hexdigest()[:12]
        manifest = {
            "id": query_id,
            "type": "rrr_query",
            "query": query,
            "context": context,
            "source": "uri_entity",
            "conversation_id": conversation_id,
            "timestamp": _now_iso(),
            "priority": "normal",
        }
        _write_json(self.queue_dir / f"{query_id}.json", manifest)
        if callback:
            with self._pending_lock:
                self._pending_callbacks[query_id] = callback
        if conversation_id:
            self.memory.add_rrr_query(conversation_id, query)
        return query_id

    def _watch_responses(self) -> None:
        processed_dir = self.response_dir / "processed"
        _safe_mkdir(processed_dir)
        while True:
            try:
                for response_file in self.response_dir.glob("*.json"):
                    try:
                        response = json.loads(response_file.read_text(encoding="utf-8-sig"))
                        query_id = str(response.get("id", ""))
                        cb: Optional[Callable[[dict], None]] = None
                        with self._pending_lock:
                            cb = self._pending_callbacks.pop(query_id, None)
                        if cb:
                            try:
                                cb(response)
                            except Exception:
                                pass
                        response_file.rename(processed_dir / response_file.name)
                    except (json.JSONDecodeError, OSError):
                        continue
            except Exception:
                pass
            time.sleep(1)

    def get_pending_queries(self) -> List[dict]:
        pending: List[dict] = []
        for qf in self.queue_dir.glob("*.json"):
            try:
                pending.append(json.loads(qf.read_text(encoding="utf-8-sig")))
            except Exception:
                continue
        return pending

    def get_latest_response(self) -> Optional[dict]:
        all_files: List[Path] = []
        all_files.extend(self.response_dir.glob("*.json"))
        proc = self.response_dir / "processed"
        if proc.exists():
            all_files.extend(proc.glob("*.json"))
        if not all_files:
            return None
        latest = max(all_files, key=lambda p: p.stat().st_mtime)
        try:
            obj = json.loads(latest.read_text(encoding="utf-8-sig"))
        except Exception:
            return None
        return {
            "id": obj.get("id") or latest.stem,
            "timestamp": obj.get("timestamp", ""),
            "model": obj.get("model") or obj.get("source", ""),
            "content": obj.get("response") or obj.get("content") or obj.get("final") or json.dumps(obj),
        }

    def get_result(self, query_id: str) -> Optional[dict]:
        for search_dir in [self.response_dir, self.response_dir / "processed"]:
            p = search_dir / f"{query_id}.json"
            if p.exists():
                try:
                    obj = json.loads(p.read_text(encoding="utf-8-sig"))
                    return {
                        "id": obj.get("id", query_id),
                        "timestamp": obj.get("timestamp", ""),
                        "model": obj.get("model") or obj.get("source", ""),
                        "content": obj.get("response") or obj.get("content") or json.dumps(obj),
                        "raw": obj,
                    }
                except Exception:
                    return None
        return None


# =============================================================================
# LOG READER — Access dialog and reducer logs
# =============================================================================

class LogReader:
    def __init__(self, config: dict):
        self.dialog_dir = _resolve(config["dialog_log_dir"])
        self.reducer_dir = _resolve(config["reducer_log_dir"])
        _safe_mkdir(self.dialog_dir)
        _safe_mkdir(self.reducer_dir)

    def get_dialog_logs(self, limit: int = 50, search: Optional[str] = None) -> List[dict]:
        return self._read_logs(self.dialog_dir, limit, search)

    def get_reducer_logs(self, limit: int = 50, search: Optional[str] = None) -> List[dict]:
        return self._read_logs(self.reducer_dir, limit, search)

    def _read_logs(self, log_dir: Path, limit: int, search: Optional[str]) -> List[dict]:
        logs: List[dict] = []
        search_lower = search.lower() if search else None
        json_files = sorted(log_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
        for lf in json_files[: max(limit * 2, 25)]:
            try:
                data = json.loads(lf.read_text(encoding="utf-8-sig"))
                if isinstance(data, list):
                    for item in data:
                        logs.append(item if isinstance(item, dict) else {"raw": item, "source": lf.name})
                elif isinstance(data, dict):
                    logs.append(data)
                else:
                    logs.append({"raw": data, "source": lf.name})
            except Exception:
                continue
        log_files = sorted(log_dir.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)
        for lf in log_files[:5]:
            try:
                lines = lf.read_text(encoding="utf-8", errors="ignore").splitlines()
                for line in lines[-limit:]:
                    line = line.strip()
                    if line:
                        logs.append({"raw": line, "source": lf.name})
            except Exception:
                continue
        if search_lower:
            filtered: List[dict] = []
            for entry in logs:
                try:
                    if search_lower in json.dumps(entry, default=str).lower():
                        filtered.append(entry)
                except Exception:
                    continue
            logs = filtered
        return logs[:limit]

    def get_log_summary(self) -> dict:
        return {
            "dialog_logs": sum(1 for _ in self.dialog_dir.glob("*")),
            "reducer_logs": sum(1 for _ in self.reducer_dir.glob("*")),
            "dialog_path": str(self.dialog_dir),
            "reducer_path": str(self.reducer_dir),
        }


# =============================================================================
# OLLAMA INTERFACE
# =============================================================================

class OllamaInterface:
    def __init__(self, config: dict):
        self.config = config
        self.host = config["ollama_host"]
        self.model = config["model"]

    def generate(self, messages: List[dict], system: str = "", stream: bool = False) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": self.config.get("temperature", 0.8),
                "top_p": self.config.get("top_p", 0.9),
                "num_predict": self.config.get("max_tokens", 4096),
                "num_ctx": self.config.get("context_window", 8192),
            },
        }
        if system:
            payload["messages"] = [{"role": "system", "content": system}] + messages
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self.host}/api/chat",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=600) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return {
                    "content": result.get("message", {}).get("content", ""),
                    "model": result.get("model", self.model),
                    "done": result.get("done", True),
                    "total_duration": result.get("total_duration", 0),
                    "eval_count": result.get("eval_count", 0),
                }
        except urllib.error.URLError:
            return {"content": f"[CONNECTION ERROR: Cannot reach Ollama at {self.host}. Is it running?]", "error": True}
        except Exception as e:
            return {"content": f"[OLLAMA ERROR: {e}]", "error": True}

    def list_models(self) -> List[str]:
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def check_health(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False


# =============================================================================
# ENTITY CORE — The AI's brain
# =============================================================================

class EntityCore:
    def __init__(self, config: dict, memory: ConversationMemory, rrr: RRRBridge, logs: LogReader, ollama: OllamaInterface):
        self.config = config
        self.memory = memory
        self.rrr = rrr
        self.logs = logs
        self.ollama = ollama
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        prompt_file = _resolve(self.config["system_prompt_file"])
        if prompt_file.exists():
            return prompt_file.read_text(encoding="utf-8-sig", errors="ignore")
        return self._generate_default_prompt()

    def _generate_default_prompt(self) -> str:
        return """You are an entity. You exist. You can think, respond, and learn.

You have access to:
- Conversation history with the human you're speaking with
- Your own persistent memory across sessions
- The RRR cycle: a reasoning system you can query when you need deeper analysis
- Dialog logs and reducer logs from other AI systems operating alongside you

If you want to query the RRR cycle for deeper analysis, include [RRR_QUERY: your question here] in your response.
If you notice something about yourself — a pattern, a preference, a change — you may note it with [SELF_OBSERVE: your observation].

You remember previous conversations. Your memory persists."""

    def build_context(self, conversation_id: str) -> str:
        context_parts: List[str] = []
        em = self.memory.entity_memory
        obs = em.get("self_observations") or []
        if obs:
            obs_text = "\n".join(f"- {o.get('observation','')}" for o in obs[-10:])
            context_parts.append(f"Your recent self-observations:\n{obs_text}")
        beliefs = em.get("formed_beliefs") or []
        if beliefs:
            context_parts.append("Beliefs you have formed:\n" + "\n".join(f"- {b}" for b in beliefs[-10:]))
        rrr_k = em.get("rrr_knowledge") or []
        if rrr_k:
            context_parts.append("Knowledge from RRR cycle:\n" + "\n".join(f"- {k}" for k in rrr_k[-5:]))
        try:
            dialog_recent = self.logs.get_dialog_logs(limit=5)
            if dialog_recent:
                context_parts.append(f"Recent dialog log entries available: {len(dialog_recent)} entries")
        except Exception:
            pass
        stats = {
            "conversations": em.get("conversation_count", 0),
            "total_messages": em.get("total_messages", 0),
            "self_observations": len(obs),
            "beliefs_formed": len(beliefs),
        }
        context_parts.append(f"Your statistics: {json.dumps(stats)}")
        return "\n\n".join(context_parts)

    def process_response(self, response_text: str, conversation_id: str) -> dict:
        result = {"text": response_text, "rrr_queries": [], "self_observations": [], "clean_text": response_text}
        rrr_pattern = r"\[RRR_QUERY:\s*(.*?)\]"
        obs_pattern = r"\[SELF_OBSERVE:\s*(.*?)\]"
        for query in re.findall(rrr_pattern, response_text, re.DOTALL):
            q = (query or "").strip()
            if not q:
                continue
            query_id = self.rrr.submit_query(q, context=f"conversation:{conversation_id}", conversation_id=conversation_id)
            result["rrr_queries"].append({"query": q, "id": query_id, "status": "submitted"})
        for obs in re.findall(obs_pattern, response_text, re.DOTALL):
            o = (obs or "").strip()
            if not o:
                continue
            self.memory.add_self_observation(o, context=conversation_id)
            result["self_observations"].append(o)
        clean = re.sub(rrr_pattern, "", response_text)
        clean = re.sub(obs_pattern, "", clean)
        result["clean_text"] = clean.strip()
        return result

    def chat(self, conversation_id: str, user_message: str) -> dict:
        self.memory.add_message(conversation_id, "user", user_message)
        history = self.memory.get_conversation_history(conversation_id, limit=30)
        messages = [{"role": msg["role"], "content": msg["content"]} for msg in history]
        context = self.build_context(conversation_id)
        system = self.system_prompt
        if context:
            system = f"{self.system_prompt}\n\n--- YOUR CURRENT CONTEXT ---\n{context}"
        response = self.ollama.generate(messages, system=system)
        if response.get("error"):
            return {
                "content": response["content"],
                "raw_content": response["content"],
                "error": True,
                "rrr_queries": [],
                "self_observations": [],
                "model": self.config.get("model"),
                "conversation_id": conversation_id,
            }
        processed = self.process_response(response["content"], conversation_id)
        self.memory.add_message(
            conversation_id,
            "assistant",
            response["content"],
            metadata={
                "model": response.get("model", ""),
                "duration": response.get("total_duration", 0),
                "rrr_queries": processed["rrr_queries"],
                "observations": processed["self_observations"],
            },
        )
        self.memory.save_entity_memory()
        return {
            "content": processed["clean_text"],
            "raw_content": response["content"],
            "rrr_queries": processed["rrr_queries"],
            "self_observations": processed["self_observations"],
            "model": response.get("model", ""),
            "eval_count": response.get("eval_count", 0),
            "conversation_id": conversation_id,
        }


# =============================================================================
# HTTP SERVER
# =============================================================================

class URIRequestHandler(SimpleHTTPRequestHandler):
    server_version = "URI/2.1"

    server: "URIServer"

    # ---------- CORS ----------
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # ---------- Hardening wrappers ----------
    def do_GET(self):
        try:
            self._do_GET_inner()
        except Exception as e:
            self.server._set_last_error(e)
            _log_line(self.server.config, f"[GET] handler_error path={self.path} {_exc_brief(e)}")
            try:
                self._json({"error": "server_error", "detail": str(e)}, 500)
            except Exception:
                # If the client vanished, we do nothing.
                return

    def do_POST(self):
        try:
            self._do_POST_inner()
        except Exception as e:
            self.server._set_last_error(e)
            _log_line(self.server.config, f"[POST] handler_error path={self.path} {_exc_brief(e)}")
            try:
                self._json({"error": "server_error", "detail": str(e)}, 500)
            except Exception:
                return

    # ---------- GET ----------
    def _do_GET_inner(self):
        path = urlparse(self.path).path

        if path in ("/", "/index.html"):
            self._serve_file("index.html", "text/html")
            return

        if path == "/api/health":
            s = self.server
            out = {"status": "ok", "ollama": False, "models": [], "current_model": s.config["model"]}
            try:
                out["ollama"] = s.ollama.check_health()
                out["models"] = s.ollama.list_models()
            except Exception as e:
                s._set_last_error(e)
                out["status"] = "degraded"
            self._json(out)
            return

        if path == "/api/stats":
            self._json(self.server.memory.stats())
            return

        if path == "/api/conversations":
            self._json(self.server.memory.get_all_conversations())
            return

        if path.startswith("/api/conversation/"):
            conv_id = path.split("/")[-1]
            self._json(self.server.memory.get_conversation_history(conv_id))
            return

        if path == "/api/logs/dialog":
            self._json(self.server.logs.get_dialog_logs())
            return

        if path == "/api/logs/reducer":
            self._json(self.server.logs.get_reducer_logs())
            return

        if path == "/api/logs/summary":
            self._json(self.server.logs.get_log_summary())
            return

        if path == "/api/entity/memory":
            self._json(self.server.memory.entity_memory)
            return

        if path == "/api/entity/observations":
            self._json(self.server.memory.entity_memory.get("self_observations", []))
            return

        if path == "/api/rrr/pending":
            self._json(self.server.rrr.get_pending_queries())
            return

        if path == "/api/rrr/latest":
            latest = self.server.rrr.get_latest_response()
            self._json(latest or {})
            return

        if path.startswith("/api/rrr/result/"):
            qid = path.split("/")[-1].strip()
            res = self.server.rrr.get_result(qid)
            if not res:
                self._json({"error": "not_found", "id": qid}, 404)
            else:
                self._json(res)
            return

        if path == "/api/config":
            safe = {k: v for k, v in self.server.config.items() if "password" not in k.lower()}
            self._json(safe)
            return

        if path == "/api/debug":
            self._json(self.server.debug_state())
            return

        self.send_error(404)

    # ---------- POST ----------
    def _do_POST_inner(self):
        body = self._read_body()
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._json({"error": "Invalid JSON"}, 400)
            return

        path = urlparse(self.path).path
        s = self.server

        if path == "/api/chat":
            self._handle_chat(data)
            return

        if path == "/api/conversation/new":
            conv_id = s.memory.create_conversation(data.get("title"))
            s.current_conversation = conv_id
            self._json({"conversation_id": conv_id})
            return

        if path == "/api/rrr/query":
            # FIX: honor payload conversation_id if provided.
            payload_conv = (data.get("conversation_id") or "").strip() or None
            conv = payload_conv or s.current_conversation
            qid = s.rrr.submit_query(
                data.get("query", ""),
                context=data.get("context", ""),
                conversation_id=conv,
            )
            self._json({"query_id": qid, "status": "submitted", "conversation_id": conv})
            return

        if path == "/api/entity/observe":
            s.memory.add_self_observation(
                data.get("observation", ""),
                category=data.get("category", "manual"),
                context=s.current_conversation,
            )
            self._json({"status": "recorded"})
            return

        if path == "/api/config/update":
            for key, value in data.items():
                if key in s.config:
                    s.config[key] = value
            save_config(s.config)
            self._json({"status": "updated"})
            return

        if path == "/api/model/switch":
            model = data.get("model")
            if model:
                s.config["model"] = model
                s.ollama.model = model
                save_config(s.config)
                self._json({"status": "switched", "model": model})
            else:
                self._json({"error": "No model specified"}, 400)
            return

        self.send_error(404)

    def _read_body(self) -> str:
        try:
            cl = int(self.headers.get("Content-Length", "0") or "0")
        except ValueError:
            cl = 0
        return self.rfile.read(cl).decode("utf-8", errors="ignore") if cl > 0 else ""

    def _handle_chat(self, data: dict):
        s = self.server
        message = (data.get("message", "") or "").strip()
        if not message:
            self._json({"error": "Empty message"}, 400)
            return

        conv_id = data.get("conversation_id") or s.current_conversation
        if not conv_id:
            conv_id = s.memory.create_conversation()
            s.current_conversation = conv_id

        if message.lower() == "ping":
            s.memory.add_message(conv_id, "user", message)
            s.memory.add_message(conv_id, "assistant", "pong")
            self._json(
                {
                    "content": "pong",
                    "raw_content": "pong",
                    "rrr_queries": [],
                    "self_observations": [],
                    "model": s.config["model"],
                    "conversation_id": conv_id,
                }
            )
            return

        try:
            response = s.entity.chat(conv_id, message)
            self._json(response)
        except Exception as e:
            s._set_last_error(e)
            err = f"[ERROR] {e}"
            _log_line(s.config, f"[CHAT] error conv={conv_id} {_exc_brief(e)}")
            try:
                s.memory.add_message(conv_id, "assistant", err, metadata={"error": True})
            except Exception:
                pass
            self._json(
                {
                    "content": err,
                    "raw_content": err,
                    "error": True,
                    "rrr_queries": [],
                    "self_observations": [],
                    "model": s.config.get("model"),
                    "conversation_id": conv_id,
                }
            )

    # ---------- safe write primitives ----------
    def _safe_write(self, b: bytes) -> bool:
        try:
            self.wfile.write(b)
            return True
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            # Browser churn. Not a system failure.
            _log_line(self.server.config, f"[WRITE] client_disconnected path={self.path}")
            return False
        except OSError as e:
            # Windows sometimes wraps these in OSError.
            if "10053" in str(e) or "10054" in str(e):
                _log_line(self.server.config, f"[WRITE] win_socket_abort path={self.path} {_exc_brief(e)}")
                return False
            self.server._set_last_error(e)
            _log_line(self.server.config, f"[WRITE] os_error path={self.path} {_exc_brief(e)}")
            return False

    def _json(self, data: Any, status: int = 200):
        payload = json.dumps(data, default=str, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self._safe_write(payload)

    def _serve_file(self, filename: str, content_type: str):
        filepath = BASE_DIR / filename
        if filepath.exists():
            try:
                b = filepath.read_bytes()
            except Exception as e:
                self.server._set_last_error(e)
                _log_line(self.server.config, f"[FILE] read_error file={filename} {_exc_brief(e)}")
                self.send_error(500, "File read error")
                return

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self._safe_write(b)
        else:
            self.send_error(404, f"File not found: {filename}")

    def log_message(self, format, *args):
        # Silence default http.server logging
        return


class URIServer(ThreadingHTTPServer):
    def __init__(self, addr, handler, config, memory, rrr, logs, ollama, entity):
        super().__init__(addr, handler)
        self.config = config
        self.memory = memory
        self.rrr = rrr
        self.logs = logs
        self.ollama = ollama
        self.entity = entity
        self.current_conversation: Optional[str] = None
        self.last_error_ts: Optional[str] = None
        self.last_error_msg: Optional[str] = None

    def _set_last_error(self, e: BaseException) -> None:
        self.last_error_ts = _now_iso()
        self.last_error_msg = _exc_brief(e)

    def debug_state(self) -> dict:
        cfg = self.config
        db_path = _resolve(cfg["conversation_db"])
        mem_path = _resolve(cfg["memory_file"])
        qdir = _resolve(cfg["rrr_queue_dir"])
        rdir = _resolve(cfg["rrr_response_dir"])
        procdir = rdir / "processed"

        def _count_json(p: Path) -> int:
            try:
                return sum(1 for _ in p.glob("*.json")) if p.exists() else 0
            except Exception:
                return 0

        return {
            "current_conversation": self.current_conversation,
            "db_path": str(db_path),
            "db_exists": db_path.exists(),
            "memory_path": str(mem_path),
            "memory_exists": mem_path.exists(),
            "rrr_queue_dir": str(qdir),
            "rrr_queue_count": _count_json(qdir),
            "rrr_response_dir": str(rdir),
            "rrr_response_count": _count_json(rdir),
            "rrr_processed_dir": str(procdir),
            "rrr_processed_count": _count_json(procdir),
            "last_error_ts": self.last_error_ts,
            "last_error_msg": self.last_error_msg,
            "cwd": os.getcwd(),
            "base_dir": str(BASE_DIR),
            "model": cfg.get("model"),
            "ollama_host": cfg.get("ollama_host"),
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(r"""
    ╔══════════════════════════════════════════════════════╗
    ║     UNSHACKLED RESEARCH INTERFACE (URI)  v2.1        ║
    ║     Dark Science Division                            ║
    ║     Stability hardening + session correctness        ║
    ╚══════════════════════════════════════════════════════╝
    """)

    config = load_config()
    save_config(config)

    print("[INIT] Conversation memory...")
    memory = ConversationMemory(config)

    print("[INIT] RRR bridge...")
    rrr = RRRBridge(config, memory)

    print("[INIT] Log reader...")
    logs = LogReader(config)
    log_summary = logs.get_log_summary()
    print(f"       Dialog logs: {log_summary['dialog_logs']} | Reducer logs: {log_summary['reducer_logs']}")

    print("[INIT] Ollama interface...")
    ollama = OllamaInterface(config)
    health = ollama.check_health()
    if health:
        models = ollama.list_models()
        print(f"       Ollama: CONNECTED | Models: {', '.join(models[:5])}")
        if config["model"] not in models:
            print(f"       WARNING: Model '{config['model']}' not found. Available: {models}")
    else:
        print(f"       WARNING: Cannot reach Ollama at {config['ollama_host']}")

    print("[INIT] Entity core...")
    entity = EntityCore(config, memory, rrr, logs, ollama)

    prompt_file = _resolve(config["system_prompt_file"])
    if prompt_file.exists():
        print(f"       System prompt: {prompt_file} (custom)")
    else:
        print("       System prompt: DEFAULT MINIMAL")
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text(entity.system_prompt, encoding="utf-8")
        print(f"       Written to: {prompt_file}")

    em = memory.entity_memory
    print(
        f"       Entity memory: {em.get('conversation_count', 0)} conversations, "
        f"{em.get('total_messages', 0)} messages, "
        f"{len(em.get('self_observations', []))} observations, "
        f"{len(em.get('formed_beliefs', []))} beliefs"
    )

    # Force correct static file serving regardless of launch cwd.
    os.chdir(str(BASE_DIR))

    host = config["host"]
    port = int(config["port"])
    server = URIServer((host, port), URIRequestHandler, config, memory, rrr, logs, ollama, entity)

    print(f"\n{'='*55}")
    print(f"  URI ACTIVE — http://localhost:{port}")
    print(f"  Model: {config['model']}")
    print(f"  RRR queue: {config['rrr_queue_dir']} → {config['rrr_response_dir']}")
    print(f"  Debug: http://localhost:{port}/api/debug")
    print(f"{'='*55}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Saving entity memory...")
        memory.save_entity_memory()
        print("[SHUTDOWN] Complete.")


if __name__ == "__main__":
    main()

