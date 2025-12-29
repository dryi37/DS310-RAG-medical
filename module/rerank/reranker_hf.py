from __future__ import annotations
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class HFCrossEncoderReranker:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_length: int = 256,
        batch_size: int = 32,
    ):
        use_cuda = torch.cuda.is_available() and device.startswith("cuda")
        self.device = "cuda" if use_cuda else "cpu"

        self.tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()

        self.max_length = max_length
        self.batch_size = batch_size

    @torch.no_grad()
    def rerank(self, query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not docs:
            return docs

        texts = [d["text"] for d in docs]
        scores = []

        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start:start + self.batch_size]
            enc = self.tok(
                [query] * len(chunk),
                chunk,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            logits = self.model(**enc).logits.squeeze(-1)
            scores.extend(logits.float().tolist())

        out = []
        for d, s in zip(docs, scores):
            d2 = dict(d)
            d2["rerank_score"] = float(s)
            out.append(d2)

        out.sort(key=lambda x: x["rerank_score"], reverse=True)
        for i, d in enumerate(out):
            d["rerank_rank"] = i + 1

        return out
