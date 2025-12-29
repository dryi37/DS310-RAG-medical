from __future__ import annotations
import time
from typing import Dict, Any

from module.router.router import GeminiSimpleRouter
from module.rewrite.query_rewrite import GeminiQueryRewriter
from module.retrival.retrival_chroma import ChromaRetriever
from module.rerank.reranker_hf import HFCrossEncoderReranker
from module.generate.generator import GeminiAnswerGenerator


class RAGPipeline:
    def __init__(
        self,
        chroma_dir: str,
        collection_name: str,
        embed_model_path: str,
        rerank_model_path: str,
        device: str = "cuda",
    ):
        self.router = GeminiSimpleRouter()
        self.rewriter = GeminiQueryRewriter(api_key=None, model="gemini-2.5-flash")  # dùng GEMINI_API_KEY trong .env

        self.retriever = ChromaRetriever(
            persist_dir=chroma_dir,
            collection_name=collection_name,
            embed_model_path=embed_model_path,
            device=device,
            use_e5_prefix=True,
        )

        self.reranker = HFCrossEncoderReranker(
            model_path=rerank_model_path,
            device=device,
            max_length=256,
            batch_size=32,
        )

        self.generator = GeminiAnswerGenerator(model="gemini-2.5-flash")

    def chat(self, conversation_id: str, query: str) -> Dict[str, Any]:
        t0 = time.time()

        route = self.router.route(query)

        # smalltalk -> trả lời nhanh
        if route == "smalltalk":
            return {
                "conversation_id": conversation_id,
                "route": "smalltalk",
                "rewritten_query": query,
                "top_docs": [],
                "answer": "Chào bạn 👋 Bạn muốn hỏi vấn đề sức khỏe nào ạ?",
                "latency_ms": int((time.time() - t0) * 1000),
            }

        # rag
        rw = self.rewriter.rewrite(query)
        rewritten = getattr(rw, "rewritten_query", query)

        docs = self.retriever.retrieve(rewritten, top_k=20)
        reranked = self.reranker.rerank(rewritten, docs)

        answer = self.generator.generate(query, reranked, max_docs=5)

        return {
            "conversation_id": conversation_id,
            "route": "rag",
            "rewritten_query": rewritten,
            "top_docs": [
                {"rank": d.get("rerank_rank", d["rank"]), "score": d.get("rerank_score", d["score"]), "text": d["text"][:600]}
                for d in reranked[:5]
            ],
            "answer": answer,
            "latency_ms": int((time.time() - t0) * 1000),
        }


if __name__ == "__main__":
    pipe = RAGPipeline(
        chroma_dir="./chroma_db",
        collection_name="medical_kb",
        embed_model_path="checkpoints/retrival/e5_small/best_model",
        rerank_model_path="checkpoints/rerank/bge_base/best_model",
        device="cuda",
    )

    res = pipe.chat("demo", "bé 17 tháng tiểu cầu có 2000 gây xuất huyết da điều trị như thế nào?")
    print(res["route"])
    print(res["rewritten_query"])
    print(res["answer"])
