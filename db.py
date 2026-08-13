"""SQLite persistence layer: conversations, messages, uploaded documents and their chunks."""

import os
import sqlite3
import time
import uuid

from typing import Optional

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chatbot.db")

_override_path: Optional[str] = None
_conns: dict[str, sqlite3.Connection] = {}


def set_db_path(path: str) -> None:
    """Override the database file (used by tests)."""
    global _override_path
    _override_path = path


def _db_path() -> str:
    return _override_path or os.getenv("DB_PATH") or DEFAULT_DB_PATH


def get_conn() -> sqlite3.Connection:
    path = _db_path()
    if path not in _conns:
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        _init_schema(conn)
        _conns[path] = conn
    return _conns[path]


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id            TEXT PRIMARY KEY,
            title         TEXT NOT NULL,
            system_prompt TEXT,
            created_at    REAL NOT NULL,
            updated_at    REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            conv_id    TEXT NOT NULL,
            role       TEXT NOT NULL,
            content    TEXT NOT NULL,
            model      TEXT,
            created_at REAL NOT NULL,
            FOREIGN KEY (conv_id) REFERENCES conversations(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS documents (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id     INTEGER NOT NULL,
            content    TEXT NOT NULL,
            embedding  BLOB,
            FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
        );
        """
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------
def create_conversation(system_prompt: Optional[str] = None) -> sqlite3.Row:
    conv_id = uuid.uuid4().hex
    now = time.time()
    conn = get_conn()
    conn.execute(
        "INSERT INTO conversations (id, title, system_prompt, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (conv_id, "New chat", system_prompt, now, now),
    )
    conn.commit()
    return get_conversation(conv_id)


def get_conversation(conv_id: str) -> Optional[sqlite3.Row]:
    row = get_conn().execute(
        "SELECT * FROM conversations WHERE id = ?", (conv_id,)
    ).fetchone()
    return row


def list_conversations() -> list[sqlite3.Row]:
    return get_conn().execute(
        "SELECT * FROM conversations ORDER BY updated_at DESC"
    ).fetchall()


def rename_conversation(conv_id: str, title: str) -> None:
    conn = get_conn()
    conn.execute(
        "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
        (title[:60], time.time(), conv_id),
    )
    conn.commit()


def set_system_prompt(conv_id: str, system_prompt: str) -> None:
    conn = get_conn()
    conn.execute(
        "UPDATE conversations SET system_prompt = ?, updated_at = ? WHERE id = ?",
        (system_prompt, time.time(), conv_id),
    )
    conn.commit()


def delete_conversation(conv_id: str) -> None:
    conn = get_conn()
    conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    conn.commit()


def add_message(conv_id: str, role: str, content: str, model: Optional[str] = None) -> None:
    conn = get_conn()
    conn.execute(
        "INSERT INTO messages (conv_id, role, content, model, created_at) VALUES (?, ?, ?, ?, ?)",
        (conv_id, role, content, model, time.time()),
    )
    conn.execute(
        "UPDATE conversations SET updated_at = ? WHERE id = ?", (time.time(), conv_id)
    )
    conn.commit()


def get_messages(conv_id: str) -> list[sqlite3.Row]:
    return get_conn().execute(
        "SELECT role, content, model FROM messages WHERE conv_id = ? ORDER BY id",
        (conv_id,),
    ).fetchall()


def count_user_messages(conv_id: str) -> int:
    row = get_conn().execute(
        "SELECT COUNT(*) AS n FROM messages WHERE conv_id = ? AND role = 'user'",
        (conv_id,),
    ).fetchone()
    return int(row["n"]) if row else 0


# ---------------------------------------------------------------------------
# Documents (RAG)
# ---------------------------------------------------------------------------
def add_document(name: str) -> int:
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO documents (name, created_at) VALUES (?, ?)", (name, time.time())
    )
    conn.commit()
    return int(cur.lastrowid)


def list_documents() -> list[sqlite3.Row]:
    return get_conn().execute(
        """
        SELECT d.id, d.name, d.created_at, COUNT(c.id) AS chunks
        FROM documents d LEFT JOIN chunks c ON c.doc_id = d.id
        GROUP BY d.id ORDER BY d.created_at DESC
        """
    ).fetchall()


def delete_document(doc_id: int) -> None:
    conn = get_conn()
    conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
    conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()


def add_chunks(doc_id: int, chunks: list[str], embeddings: list[list[float]]) -> None:
    conn = get_conn()
    for i, chunk in enumerate(chunks):
        blob = None
        if embeddings and i < len(embeddings):
            blob = _encode_embedding(embeddings[i])
        conn.execute(
            "INSERT INTO chunks (doc_id, content, embedding) VALUES (?, ?, ?)",
            (doc_id, chunk, blob),
        )
    conn.commit()


def get_all_chunks() -> list[sqlite3.Row]:
    return get_conn().execute(
        "SELECT content, embedding FROM chunks WHERE embedding IS NOT NULL"
    ).fetchall()


def _encode_embedding(vector: list[float]) -> bytes:
    import numpy as np

    return np.asarray(vector, dtype=np.float32).tobytes()