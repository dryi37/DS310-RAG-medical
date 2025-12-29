from __future__ import annotations
from dotenv import load_dotenv
import os
import json
from dataclasses import dataclass
from typing import List

from google import genai

load_dotenv()


@dataclass
class RewriteResult:
    rewritten_query: str
    keywords: List[str]


class GeminiQueryRewriter:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def rewrite(self, user_query: str) -> RewriteResult:
        prompt = f"""
Bạn là module Query Rewrite cho hệ thống RAG y khoa.

NHIỆM VỤ:
- Viết lại câu hỏi ngắn gọn, đúng trọng tâm để truy vấn dữ liệu.
- Không chẩn đoán bệnh.
- Không thêm thông tin không có trong câu hỏi.

CHỈ TRẢ VỀ JSON, KHÔNG GIẢI THÍCH.

JSON FORMAT:
{{
  "rewritten_query": "string",
  "keywords": ["string", "..."]
}}

CÂU HỎI:
{user_query}
"""

        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        text = (resp.text or "").strip()

        # Tách JSON an toàn
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            data = json.loads(text[start:end])
        except Exception:
            # fallback an toàn
            return RewriteResult(
                rewritten_query=user_query,
                keywords=[]
            )

        return RewriteResult(
            rewritten_query=data.get("rewritten_query", user_query),
            keywords=data.get("keywords", []),
        )


if __name__ == "__main__":
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY")

    rewriter = GeminiQueryRewriter(api_key)

    q = "Khoảng hơn một tháng nay tôi thường xuyên bị đau âm ỉ vùng khớp háng bên phải, đặc biệt khi đứng lên ngồi xuống hoặc xoay người thì cảm giác đau tăng rõ rệt, đôi lúc lan xuống mông và đùi, buổi sáng ngủ dậy cảm thấy cứng khớp khó vận động, xin hỏi đây có thể là dấu hiệu của bệnh gì và tôi nên đi khám chuyên khoa nào?"
    result = rewriter.rewrite(q)

    print("Rewritten:", result.rewritten_query)
    print("Keywords:", result.keywords)
