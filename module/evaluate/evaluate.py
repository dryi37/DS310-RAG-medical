import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import json
import argparse
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# -----------------------
# Metrics
# -----------------------
def recall_at_k(ranked_ids: List[str], gold_id: str, k: int) -> float:
    return 1.0 if gold_id in ranked_ids[:k] else 0.0


def mrr(ranked_ids: List[str], gold_id: str) -> float:
    for i, did in enumerate(ranked_ids):
        if did == gold_id:
            return 1.0 / (i + 1)
    return 0.0


# -----------------------
# Dataset I/O
# -----------------------
def load_dataset(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # accept a single object as well
    if isinstance(data, dict):
        return [data]
    return data


def build_pool(sample: Dict[str, Any]):
    """
    Dataset structure (yours):
    - question
    - positive
    - negatives: list[str]
    """
    q = (sample.get("question") or "").strip()
    pos = (sample.get("positive") or "").strip()
    negs = sample.get("negatives") or []
    negs = [str(x).strip() for x in negs if str(x).strip()]

    doc_texts = [pos] + negs
    doc_ids = ["pos"] + [f"neg_{i}" for i in range(len(negs))]
    return q, doc_texts, doc_ids


# -----------------------
# Rewrite wrappers
# -----------------------
class NoRewrite:
    def rewrite(self, q: str) -> str:
        return q


class GeminiRewriteWrapper:
    """
    Tries to import your Gemini rewriter.
    Expect file: rewrite/query_rewrite.py
    class: GeminiQueryRewriter(api_key).rewrite(q) -> RewriteResult or string
    """
    def __init__(self):
        from rewrite.query_rewrite import GeminiQueryRewriter  # adjust if needed
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY in environment/.env")
        self.rw = GeminiQueryRewriter(api_key)

    def rewrite(self, q: str) -> str:
        res = self.rw.rewrite(q)
        return getattr(res, "rewritten_query", res)


# -----------------------
# Reranker (your HF cross-encoder)
# -----------------------
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
        self.max_length = max_length
        self.batch_size = batch_size

        self.tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def rerank(self, query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not docs:
            return docs

        texts = [d["text"] for d in docs]
        scores: List[float] = []

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

            logits = self.model(**enc).logits.squeeze(-1)  # (B,)
            scores.extend(logits.float().tolist())

        for d, s in zip(docs, scores):
            d["rerank_score"] = float(s)

        docs = sorted(docs, key=lambda x: x["rerank_score"], reverse=True)
        return docs


# -----------------------
# Retrieval (cosine on normalized embeddings)
# -----------------------
def retrieve_rank_ids(
    embedder: SentenceTransformer,
    query_text: str,
    doc_texts: List[str],
    doc_ids: List[str],
    top_k: int,
    use_e5_prefix: bool = True,
):
    if use_e5_prefix:
        q_in = "query: " + query_text
        d_in = ["passage: " + t for t in doc_texts]
    else:
        q_in = query_text
        d_in = doc_texts

    q_emb = embedder.encode([q_in], normalize_embeddings=True)[0]       # (D,)
    d_embs = embedder.encode(d_in, normalize_embeddings=True)           # (N,D)

    sims = d_embs @ q_emb  # cosine = dot since normalized
    order = np.argsort(-sims)

    top_k = min(top_k, len(order))
    order = order[:top_k]
    ranked_ids = [doc_ids[i] for i in order]
    ranked_docs = [{"id": doc_ids[i], "text": doc_texts[i], "retr_score": float(sims[i])} for i in order]
    return ranked_ids, ranked_docs


# -----------------------
# Evaluate one config
# -----------------------
def evaluate_config(
    name: str,
    dataset: List[Dict[str, Any]],
    embedder: SentenceTransformer,
    *,
    rewriter,
    reranker: Optional[HFCrossEncoderReranker],
    do_rewrite: bool,
    do_rerank: bool,
    top_k_retrieve: int,
    eval_ks: List[int],
    use_e5_prefix: bool,
) -> Dict[str, float]:

    recall_sums = {k: 0.0 for k in eval_ks}
    mrr_sum = 0.0
    n = 0

    for sample in dataset:
        q, doc_texts, doc_ids = build_pool(sample)
        if not q or not doc_texts:
            continue

        # rewrite optional
        q_used = q
        if do_rewrite:
            try:
                q_used = rewriter.rewrite(q)
            except Exception:
                q_used = q

        # retrieval
        ranked_ids, ranked_docs = retrieve_rank_ids(
            embedder, q_used, doc_texts, doc_ids,
            top_k=top_k_retrieve,
            use_e5_prefix=use_e5_prefix,
        )

        # rerank optional (over retrieved set)
        if do_rerank and reranker is not None:
            try:
                reranked_docs = reranker.rerank(q_used, ranked_docs)
                ranked_ids = [d["id"] for d in reranked_docs]
            except Exception:
                pass

        # metrics
        gold_id = "pos"
        for k in eval_ks:
            recall_sums[k] += recall_at_k(ranked_ids, gold_id, k=min(k, len(ranked_ids)))
        mrr_sum += mrr(ranked_ids, gold_id)
        n += 1

    if n == 0:
        return {f"Recall@{k}": 0.0 for k in eval_ks} | {"MRR": 0.0}

    out = {f"Recall@{k}": recall_sums[k] / n for k in eval_ks}
    out["MRR"] = mrr_sum / n
    return out


def print_table(results: Dict[str, Dict[str, float]]):
    # collect keys
    metric_keys = []
    for m in results.values():
        for k in m.keys():
            if k not in metric_keys:
                metric_keys.append(k)

    header = ["Pipeline"] + metric_keys
    rows = []
    for name, m in results.items():
        rows.append([name] + [f"{m.get(k, 0.0):.4f}" for k in metric_keys])

    col_widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]

    def fmt(row):
        return " | ".join(str(x).ljust(w) for x, w in zip(row, col_widths))

    print(fmt(header))
    print("-|-".join("-" * w for w in col_widths))
    for r in rows:
        print(fmt(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="test.json or test.jsonl (list of items)")
    ap.add_argument("--embed_model", required=True, help="SentenceTransformer path, e.g. checkpoints/retrival/e5_small/best_model")
    ap.add_argument("--topk", type=int, default=20, help="retrieve top-k from candidate pool")
    ap.add_argument("--use_gemini_rewrite", action="store_true", help="use your Gemini rewriter (needs GEMINI_API_KEY)")
    ap.add_argument("--rerank_model", default="", help="path to reranker best_model, empty => disable rerank configs")
    ap.add_argument("--device", default="cuda", help="cuda or cpu for reranker")
    ap.add_argument("--rerank_max_length", type=int, default=256)
    ap.add_argument("--rerank_batch", type=int, default=32)
    ap.add_argument("--no_e5_prefix", action="store_true", help="disable E5 query:/passage: prefix")
    args = ap.parse_args()

    dataset = load_dataset(args.data)
    embedder = SentenceTransformer(args.embed_model)

    # rewriter
    if args.use_gemini_rewrite:
        rewriter = GeminiRewriteWrapper()
    else:
        rewriter = NoRewrite()

    # reranker (optional)
    reranker = None
    if args.rerank_model:
        reranker = HFCrossEncoderReranker(
            model_path=args.rerank_model,
            device=args.device,
            max_length=args.rerank_max_length,
            batch_size=args.rerank_batch,
        )

    eval_ks = [1, 3, 5, 10]
    use_e5_prefix = not args.no_e5_prefix

    results = {}

    # 4 configs requested
    results["baseline"] = evaluate_config(
        "baseline", dataset, embedder,
        rewriter=rewriter, reranker=reranker,
        do_rewrite=False, do_rerank=False,
        top_k_retrieve=args.topk, eval_ks=eval_ks,
        use_e5_prefix=use_e5_prefix,
    )

    results["baseline+rewrite"] = evaluate_config(
        "baseline+rewrite", dataset, embedder,
        rewriter=rewriter, reranker=reranker,
        do_rewrite=True, do_rerank=False,
        top_k_retrieve=args.topk, eval_ks=eval_ks,
        use_e5_prefix=use_e5_prefix,
    )

    results["baseline+rerank"] = evaluate_config(
        "baseline+rerank", dataset, embedder,
        rewriter=rewriter, reranker=reranker,
        do_rewrite=False, do_rerank=(reranker is not None),
        top_k_retrieve=args.topk, eval_ks=eval_ks,
        use_e5_prefix=use_e5_prefix,
    )

    results["baseline+rewrite+rerank"] = evaluate_config(
        "baseline+rewrite+rerank", dataset, embedder,
        rewriter=rewriter, reranker=reranker,
        do_rewrite=True, do_rerank=(reranker is not None),
        top_k_retrieve=args.topk, eval_ks=eval_ks,
        use_e5_prefix=use_e5_prefix,
    )

    print_table(results)


if __name__ == "__main__":
    main()