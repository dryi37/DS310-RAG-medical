from __future__ import annotations

from dotenv import load_dotenv
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Optional

from google import genai

load_dotenv()


@dataclass
class ChatMessage:
    role: str
    content: str


class SQLiteMemoryStore:
    def __init__(self, db_path: str = "memory.db"):
        import sqlite3
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                conversation_id TEXT,
                ts REAL,
                role TEXT,
                content TEXT
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                conversation_id TEXT PRIMARY KEY,
                summary TEXT,
                updated_ts REAL
            )
        """)
        self.conn.commit()

    def add_message(self, conversation_id: str, role: str, content: str):
        import time
        self.conn.execute(
            "INSERT INTO messages VALUES (?, ?, ?, ?)",
            (conversation_id, time.time(), role, content),
        )
        self.conn.commit()

    def get_recent_messages(self, conversation_id: str, limit: int = 12):
        cur = self.conn.execute(
            "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY ts DESC LIMIT ?",
            (conversation_id, limit),
        )
        rows = cur.fetchall()[::-1]
        return [{"role": r, "content": c} for r, c in rows]

    def get_summary(self, conversation_id: str) -> Optional[str]:
        cur = self.conn.execute(
            "SELECT summary FROM summaries WHERE conversation_id=?",
            (conversation_id,),
        )
        row = cur.fetchone()
        return row[0] if row else None

    def save_summary(self, conversation_id: str, summary: str):
        import time
        self.conn.execute(
            """
            INSERT INTO summaries (conversation_id, summary, updated_ts)
            VALUES (?, ?, ?)
            ON CONFLICT(conversation_id)
            DO UPDATE SET summary=excluded.summary, updated_ts=excluded.updated_ts
            """,
            (conversation_id, summary, time.time()),
        )
        self.conn.commit()


# ===============================
# LLM SUMMARIZER
# ===============================

class MemorySummarizer:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def summarize(self, old_summary: Optional[str], messages: List[Dict[str, str]]) -> str:
        history = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

        prompt = f"""
Bạn là module tóm tắt hội thoại cho hệ thống RAG y khoa.

YÊU CẦU:
- Chỉ tóm tắt thông tin người dùng cung cấp.
- Không suy đoán bệnh.
- Không đưa lời khuyên y khoa.
- Giữ lại triệu chứng, mốc thời gian, diễn biến, mong muốn của người dùng.

TÓM TẮT CŨ:
{old_summary or "(chưa có)"}

HỘI THOẠI MỚI:
{history}

Hãy tạo bản tóm tắt mới ngắn gọn (4–6 dòng), tiếng Việt.
"""

        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        return (resp.text or "").strip()


# ===============================
# API CHÍNH
# ===============================

class ConversationMemory:
    def __init__(self, db_path: str, api_key: str):
        self.store = SQLiteMemoryStore(db_path)
        self.summarizer = MemorySummarizer(api_key)

    def add_message(self, conversation_id: str, role: str, content: str):
        self.store.add_message(conversation_id, role, content)

    def get_context(self, conversation_id: str) -> Dict[str, str]:
        summary = self.store.get_summary(conversation_id)
        recent = self.store.get_recent_messages(conversation_id)

        return {
            "summary": summary or "",
            "recent_messages": recent,
        }

    def update_summary_if_needed(self, conversation_id: str):
        recent = self.store.get_recent_messages(conversation_id, limit=10)
        if len(recent) < 6:
            return

        old_summary = self.store.get_summary(conversation_id)
        new_summary = self.summarizer.summarize(old_summary, recent)

        if new_summary:
            self.store.save_summary(conversation_id, new_summary)
