from __future__ import annotations

import os
import json
import time
import hashlib
import sqlite3
from typing import Optional, Literal

from dotenv import load_dotenv
from google import genai

load_dotenv()

Route = Literal["smalltalk", "rag"]


# =============================
# SQLite cache
# =============================
class RouterCache:
    def __init__(self, path: str = "router_cache.db"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS route_cache (
                qhash TEXT PRIMARY KEY,
                route TEXT NOT NULL,
                updated_ts REAL NOT NULL
            )
        """)
        self.conn.commit()

    def get(self, qhash: str, ttl: int = 7 * 24 * 3600) -> Optional[str]:
        cur = self.conn.execute(
            "SELECT route, updated_ts FROM route_cache WHERE qhash=?",
            (qhash,),
        )
        row = cur.fetchone()
        if not row:
            return None
        route, ts = row
        if time.time() - ts > ttl:
            return None
        return route

    def set(self, qhash: str, route: str):
        self.conn.execute(
            """
            INSERT INTO route_cache (qhash, route, updated_ts)
            VALUES (?, ?, ?)
            ON CONFLICT(qhash)
            DO UPDATE SET route=excluded.route, updated_ts=excluded.updated_ts
            """,
            (qhash, route, time.time()),
        )
        self.conn.commit()


# =============================
# Router
# =============================
class GeminiSimpleRouter:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing GEMINI_API_KEY")

        self.client = genai.Client(api_key=self.api_key)
        self.model = model
        self.cache = RouterCache()

    def route(self, user_text: str) -> str:
        text = (user_text or "").strip()
        if not text:
            return "smalltalk"

        qhash = hashlib.sha1(text.lower().encode()).hexdigest()

        # 1. Cache
        cached = self.cache.get(qhash)
        if cached:
            return cached

        # 2. Ask LLM
        try:
            route = self._route_llm(text)
        except Exception:
            route = self._fallback(text)

        self.cache.set(qhash, route)
        return route

    def _route_llm(self, text: str) -> str:
        prompt = f"""
You are a router for a Vietnamese chatbot.

Return ONLY valid JSON.

Schema:
{{
  "route": "smalltalk" | "rag"
}}

Rules:
- smalltalk: greetings, thanks, casual chat, identity questions.
- rag: any question that requires factual or medical knowledge.

User input:
{text}
"""

        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        raw = (resp.text or "").strip()
        if raw.startswith("```"):
            raw = raw.replace("```json", "").replace("```", "").strip()

        try:
            data = json.loads(raw)
            route = data.get("route", "rag")
            if route not in ("smalltalk", "rag"):
                route = "rag"
            return route
        except Exception:
            return "rag"

    def _fallback(self, text: str) -> str:
        # fallback cực đơn giản
        if len(text.split()) <= 3:
            return "smalltalk"
        return "rag"



if __name__ == "__main__":
    r = GeminiSimpleRouter()
    for q in ["hello", "cảm ơn", "bé 17 tháng tiểu cầu 2000 thì sao?", "đau bụng"]:
        print(q, "=>", r.route(q))
