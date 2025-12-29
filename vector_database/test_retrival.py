import chromadb
from sentence_transformers import SentenceTransformer

# ===================== CONFIG =====================
MODEL_PATH = "checkpoints/retrival/e5_small/best_model"
DEVICE = "cuda"          # "cpu" nếu không có GPU
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "medical_kb"

TOP_K = 5
CHROMA_SPACE = "cosine"  # bạn set hnsw:space="cosine" lúc build DB
# =================================================

def distance_to_similarity(d: float, space: str = "cosine") -> float:
    """
    Chroma trả distance: nhỏ hơn = gần hơn.
    Với cosine distance thường là: distance = 1 - cosine_similarity
    => similarity = 1 - distance
    """
    if d is None:
        return 0.0
    if space == "cosine":
        sim = 1.0 - d
        return max(0.0, min(1.0, sim))
    return 1.0 / (1.0 + max(0.0, d))

def pretty_meta(meta: dict) -> str:
    if not meta:
        return "theme=(none) | title=(none) | url=(none) | row_index=(none)"
    return (
        f"theme={meta.get('theme','(none)')} | "
        f"title={meta.get('title','(none)')} | "
        f"url={meta.get('url','(none)')} | "
        f"row_index={meta.get('row_index','(none)')}"
    )

def main():
    # 1) Load embedding model
    embed_model = SentenceTransformer(MODEL_PATH, device=DEVICE)

    # 2) Load Chroma DB
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    print("Total docs in DB:", collection.count())

    # 3) Query test
    query = (
        "Em bị đau ở khớp háng, khi xoay nghe tiếng kêu, đang ngồi đứng dậy đột ngột bị đau "
        "như có cái gì đâm vào. Hiện nay, em hay đau tê cả háng và phần mông khi trời động, "
        "nhất là gần sáng. Vậy bác sĩ cho em hỏi đau khớp háng, khi xoay nghe tiếng kêu là bệnh gì?"
    )

    # (Optional) filter theo theme
    # where = {"theme": "Chấn thương chỉnh hình"}
    where = None

    # E5 prefix
    query_emb = embed_model.encode(
        ["query: " + query],
        normalize_embeddings=True
    )

    # 4) Search
    # IMPORTANT: Chroma version của bạn không cho include "ids"
    results = collection.query(
        query_embeddings=query_emb.tolist(),
        n_results=TOP_K,
        where=where,
        include=["documents", "distances", "metadatas"]
    )

    # 5) Print results
    print("\nQUERY:\n", query)
    print("=" * 110)

    docs = results.get("documents", [[]])[0]
    dists = results.get("distances", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    # ids thường vẫn có sẵn trong results (tuỳ version)
    ids_list = results.get("ids", [[]])
    ids = ids_list[0] if ids_list and isinstance(ids_list, list) else []

    if not docs:
        print("No results found.")
        print("Raw keys returned by chroma:", list(results.keys()))
        return

    for i in range(len(docs)):
        doc_id = ids[i] if i < len(ids) else "(no-id)"
        dist = dists[i] if i < len(dists) else None
        sim = distance_to_similarity(dist, CHROMA_SPACE)

        meta = metas[i] if (metas and i < len(metas) and metas[i] is not None) else {}
        print(f"[{i+1}] similarity={sim:.4f} | distance={dist:.4f} | id={doc_id}")
        print("     " + pretty_meta(meta))
        print("-" * 110)

        # Preview để khỏi spam
        doc_text = docs[i]
        preview = doc_text if len(doc_text) <= 1200 else doc_text[:1200] + "...\n[TRUNCATED]"
        print(preview)
        print("=" * 110)

if __name__ == "__main__":
    main()
