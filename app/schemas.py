from pydantic import BaseModel
from typing import Any, Dict, List, Optional

class ChatRequest(BaseModel):
    conversation_id: str = "demo"
    query: str

class DocItem(BaseModel):
    rank: int
    score: float
    text: str
    meta: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    conversation_id: str
    route: str
    query: str
    rewritten_query: str
    answer: str
    top_docs: List[DocItem]
    latency_ms: int
