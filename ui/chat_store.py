# chat_store.py
from __future__ import annotations
import sqlite3
import time
import uuid
from typing import List, Dict, Optional


class ChatStore:
    def __init__(self, db_path: str = "chat_ui.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_ts REAL NOT NULL,
                updated_ts REAL NOT NULL
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                conversation_id TEXT NOT NULL,
                ts REAL NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def new_conversation(self, title: str = "New chat") -> str:
        cid = str(uuid.uuid4())
        now = time.time()
        self.conn.execute(
            "INSERT INTO conversations (id, title, created_ts, updated_ts) VALUES (?,?,?,?)",
            (cid, title, now, now),
        )
        self.conn.commit()
        return cid

    def list_conversations(self, limit: int = 30) -> List[Dict]:
        cur = self.conn.execute(
            "SELECT id, title, updated_ts FROM conversations ORDER BY updated_ts DESC LIMIT ?",
            (limit,),
        )
        return [{"id": r[0], "title": r[1], "updated_ts": r[2]} for r in cur.fetchall()]

    def rename_conversation(self, cid: str, title: str) -> None:
        now = time.time()
        self.conn.execute(
            "UPDATE conversations SET title=?, updated_ts=? WHERE id=?",
            (title, now, cid),
        )
        self.conn.commit()

    def add_message(self, cid: str, role: str, content: str) -> None:
        now = time.time()
        self.conn.execute(
            "INSERT INTO messages (conversation_id, ts, role, content) VALUES (?,?,?,?)",
            (cid, now, role, content),
        )
        self.conn.execute("UPDATE conversations SET updated_ts=? WHERE id=?", (now, cid))
        self.conn.commit()

    def get_messages(self, cid: str, limit: int = 200) -> List[Dict]:
        cur = self.conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY ts ASC LIMIT ?",
            (cid, limit),
        )
        return [{"role": r[0], "content": r[1]} for r in cur.fetchall()]

    def delete_conversation(self, cid: str) -> None:
        self.conn.execute("DELETE FROM messages WHERE conversation_id=?", (cid,))
        self.conn.execute("DELETE FROM conversations WHERE id=?", (cid,))
        self.conn.commit()
