import os
import csv
import hashlib
import chromadb
from sentence_transformers import SentenceTransformer

# ========== CONFIG ==========
DATA_PATH = "data/raw_data.csv"
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "medical_kb"
MODEL_PATH = "checkpoints/retrival/e5_small/best_model"

DEVICE = "cuda"  # hoặc "cpu"
EMBED_BATCH_SIZE = 64
CHROMA_ADD_BATCH = 5000  # phải < max batch size

# Nếu True: xoá collection cũ và build lại từ đầu (khuyên dùng khi đang lỗi trùng id)
REBUILD_COLLECTION = True

# Nếu True: bỏ các dòng trùng (url,title,question) để DB gọn hơn
DEDUPLICATE = True
# ============================


def build_doc_text(row: dict) -> str:
    """Gộp theme/title/question/answer thành 1 document text."""
    theme = (row.get("theme") or "").strip()
    title = (row.get("title") or "").strip()
    question = (row.get("question") or "").strip()
    answer = (row.get("answer") or "").strip()

    parts = []
    if theme:
        parts.append(f"Chủ đề: {theme}")
    if title:
        parts.append(f"Tiêu đề: {title}")
    if question:
        parts.append(f"Câu hỏi: {question}")
    if answer:
        parts.append(f"Trả lời: {answer}")

    return "\n".join(parts).strip()


def stable_id(row: dict, row_index: int) -> str:
    """
    Tạo id đảm bảo unique.
    - Hash theo (url + title + question) để "gần ổn định"
    - Thêm row_index để KHÔNG BAO GIỜ trùng.
    """
    url = (row.get("URL") or row.get("url") or "").strip()
    title = (row.get("title") or "").strip()
    question = (row.get("question") or "").strip()

    key = (url + "||" + title + "||" + question).strip()
    if not key:
        return f"doc_{row_index}"

    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"doc_{h}_{row_index}"


def iter_csv(path: str):
    """Đọc CSV UTF-8. Dùng utf-8-sig để tránh BOM."""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def main():
    # 1) Load embed model
    embed_model = SentenceTransformer(MODEL_PATH, device=DEVICE)

    # 2) Persistent client
    os.makedirs(PERSIST_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=PERSIST_DIR)

    # 2.1) Rebuild collection nếu cần
    if REBUILD_COLLECTION:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"[INFO] Deleted existing collection: {COLLECTION_NAME}")
        except Exception:
            # collection có thể chưa tồn tại
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # 3) Đọc csv -> embed theo batch -> add theo batch
    buffer_docs, buffer_ids, buffer_metas = [], [], []
    added = 0

    # DEDUP: bỏ các record trùng url/title/question
    seen_keys = set()

    for idx, row in enumerate(iter_csv(DATA_PATH)):
        # DEDUP KEY (tuỳ chọn)
        if DEDUPLICATE:
            url = (row.get("URL") or row.get("url") or "").strip()
            title = (row.get("title") or "").strip()
            question = (row.get("question") or "").strip()
            key = (url, title, question)
            if key in seen_keys:
                continue
            seen_keys.add(key)

        doc_text = build_doc_text(row)
        if not doc_text:
            continue

        doc_id = stable_id(row, idx)

        meta = {
            "theme": (row.get("theme") or "").strip(),
            "title": (row.get("title") or "").strip(),
            "url": (row.get("URL") or row.get("url") or "").strip(),
            "row_index": idx,
        }

        buffer_docs.append(doc_text)
        buffer_ids.append(doc_id)
        buffer_metas.append(meta)

        # Add theo batch
        if len(buffer_ids) >= CHROMA_ADD_BATCH:
            try:
                embs = embed_model.encode(
                    ["passage: " + d for d in buffer_docs],
                    batch_size=EMBED_BATCH_SIZE,
                    normalize_embeddings=True,
                    show_progress_bar=True,
                )

                collection.add(
                    ids=buffer_ids,
                    documents=buffer_docs,
                    metadatas=buffer_metas,
                    embeddings=embs.tolist(),
                )

                added += len(buffer_ids)
                print(f"[ADD] {added} docs")

            except Exception as e:
                print("[ERROR] Failed at batch add.")
                print("First ID in batch:", buffer_ids[0] if buffer_ids else None)
                print("Last  ID in batch:", buffer_ids[-1] if buffer_ids else None)
                raise e
            finally:
                buffer_docs.clear()
                buffer_ids.clear()
                buffer_metas.clear()

    # 4) Flush phần còn lại
    if buffer_ids:
        try:
            embs = embed_model.encode(
                ["passage: " + d for d in buffer_docs],
                batch_size=EMBED_BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            collection.add(
                ids=buffer_ids,
                documents=buffer_docs,
                metadatas=buffer_metas,
                embeddings=embs.tolist(),
            )
            added += len(buffer_ids)
            print(f"[ADD] {added} docs")
        except Exception as e:
            print("[ERROR] Failed at final flush add.")
            raise e

    print("Done. Total docs in collection:", collection.count())
    print("DB saved at:", os.path.abspath(PERSIST_DIR))
    if DEDUPLICATE:
        print(f"[INFO] Deduplicated unique keys: {len(seen_keys)}")


if __name__ == "__main__":
    main()
