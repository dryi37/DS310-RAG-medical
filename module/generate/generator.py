from __future__ import annotations
import os
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from google import genai

load_dotenv()


class GeminiAnswerGenerator:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
    ):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY in .env")

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(self, user_query: str, docs: List[Dict[str, Any]], max_docs: int = 5) -> str:
        top = docs[:max_docs]

        context_blocks = []
        for i, d in enumerate(top, start=1):
            txt = (d.get("text") or "").strip()
            if not txt:
                continue
            context_blocks.append(f"[DOC {i}]\n{txt}")

        context = "\n\n".join(context_blocks).strip()

        prompt = f"""
Bạn là trợ lý y khoa dạng RAG.
CHỈ được dùng thông tin trong CONTEXT để trả lời.
Nếu không đủ thông tin trong CONTEXT, hãy nói rõ: "Mình chưa tìm thấy đủ thông tin trong tài liệu để trả lời chắc chắn."

Yêu cầu:
- Trả lời tiếng Việt, rõ ràng, không chẩn đoán chắc chắn.
- Cuối câu trả lời, thêm mục "Nguồn" dạng: [DOC 1], [DOC 2]... (chỉ những doc đã dùng).

CONTEXT:
{context}

CÂU HỎI:
{user_query}
""".strip()

        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        return (resp.text or "").strip()
