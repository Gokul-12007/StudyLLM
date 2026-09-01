"""Local, on-disk persistence for projects / chats / messages.

Deliberately plain sqlite3 (no ORM) so there's one obvious place to look
when debugging. All data lives in notebook.db next to the app — nothing
is kept only in memory, so closing the app never loses chat history.
"""
import sqlite3
import uuid
from datetime import datetime
from contextlib import contextmanager

DB_PATH = "notebook.db"


def _now():
    return datetime.utcnow().isoformat()


def _new_id():
    return uuid.uuid4().hex[:12]


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                md_path TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )


# --------------------------------------------------------------------- #
# Projects
# --------------------------------------------------------------------- #
def create_project(name):
    pid = _new_id()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO projects (id, name, created_at) VALUES (?, ?, ?)",
            (pid, name, _now()),
        )
    return pid


def list_projects():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM projects ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------- #
def add_document(project_id, filename, file_path, md_path):
    did = _new_id()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO documents (id, project_id, filename, file_path, md_path, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (did, project_id, filename, file_path, md_path, _now()),
        )
    return did


def list_documents(project_id):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM documents WHERE project_id = ? ORDER BY created_at",
            (project_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------- #
# Chats
# --------------------------------------------------------------------- #
def create_chat(project_id, title="New chat"):
    cid = _new_id()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO chats (id, project_id, title, created_at) VALUES (?, ?, ?, ?)",
            (cid, project_id, title, _now()),
        )
    return cid


def list_chats(project_id):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM chats WHERE project_id = ? ORDER BY created_at DESC",
            (project_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def rename_chat(chat_id, title):
    with get_conn() as conn:
        conn.execute("UPDATE chats SET title = ? WHERE id = ?", (title, chat_id))


# --------------------------------------------------------------------- #
# Messages
# --------------------------------------------------------------------- #
def add_message(chat_id, role, content):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO messages (chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (chat_id, role, content, _now()),
        )


def list_messages(chat_id):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE chat_id = ? ORDER BY id",
            (chat_id,),
        ).fetchall()
    return [dict(r) for r in rows]