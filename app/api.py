from __future__ import annotations
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import ChatRequest, ChatResponse
from module.pipeline_RAG import RAGPipeline

app = FastAPI(title="Medical RAG Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo local, prod thì giới hạn domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo pipeline 1 lần
pipe = RAGPipeline(
    chroma_dir="./chroma_db",
    collection_name="medical_kb",
    embed_model_path="checkpoints/retrival/e5_small/best_model",
    rerank_model_path="checkpoints/rerank/bge_base/best_model",
    device="cuda",
)

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    t0 = time.time()
    res = pipe.chat(req.conversation_id, req.query)
    res["query"] = req.query
    res["latency_ms"] = int((time.time() - t0) * 1000)
    return res
