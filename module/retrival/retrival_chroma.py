from __future__ import annotations
from typing import List, Dict, Any, Optional

import chromadb
from sentence_transformers import SentenceTransformer


class ChromaRetriever:
    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "medical_kb",
        embed_model_path: str = "checkpoints/retrival/e5_small/best_model",
        device: str = "cuda",
        use_e5_prefix: bool = True,
    ):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_collection(collection_name)

        self.embedder = SentenceTransformer(embed_model_path, device=device)
        self.use_e5_prefix = use_e5_prefix

    def retrieve(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        q = query.strip()
        if self.use_e5_prefix:
            q = "query: " + q

        q_emb = self.embedder.encode([q], normalize_embeddings=True)

        res = self.collection.query(
            query_embeddings=q_emb.tolist(),
            n_results=top_k,
            include=["documents", "metadatas", "distances"],  # CHROMA KHÔNG có "ids" trong include
        )

        docs = []
        for i, doc_text in enumerate(res["documents"][0]):
            dist = float(res["distances"][0][i])  # cosine distance: nhỏ là gần
            sim = 1.0 - dist                       # đổi sang similarity để dễ hiểu
            meta = (res["metadatas"][0][i] if res.get("metadatas") else None) or {}

            docs.append({
                "rank": i + 1,
                "text": doc_text,
                "distance": dist,
                "score": sim,
                "meta": meta,
            })

        return docs
