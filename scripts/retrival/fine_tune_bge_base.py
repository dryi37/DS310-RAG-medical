import argparse
import os
import json
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

os.environ["WANDB_DISABLED"] = "true"


def load_train_data(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            q = item["question"].strip()
            pos = item["positive"].strip()
            if q and pos:
                data.append(InputExample(texts=[q, pos]))
    return data

def load_val_data(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            q = item["question"].strip()
            pos = item["positive"].strip()
            if q and pos:
                data.append({"question": q, "positive": pos})
    return data


def evaluate_retriever(model, val_data, ks=(5, 10)):
    model.eval()

    questions = [it["question"] for it in val_data]
    answers = [it["positive"] for it in val_data]

    # Map answer text -> list of indices (handle duplicate answers)
    ans_to_indices = defaultdict(list)
    for idx, ans in enumerate(answers):
        ans_to_indices[ans].append(idx)

    # Encode on device as tensors
    with torch.no_grad():
        q_emb = model.encode(questions, convert_to_tensor=True, batch_size=32)
        a_emb = model.encode(answers, convert_to_tensor=True, batch_size=32)

    # Normalize so cosine = dot product
    q_emb = F.normalize(q_emb, p=2, dim=1)
    a_emb = F.normalize(a_emb, p=2, dim=1)

    # Similarity matrix (num_q, num_a)
    sims = torch.matmul(q_emb, a_emb.T)

    total = sims.size(0)
    recall_counts = {k: 0 for k in ks}
    mrr_sum = 0.0

    for i in range(total):
        row = sims[i]
        # Sort all candidates once
        ranks = torch.argsort(row, descending=True)

        gold_indices = ans_to_indices[answers[i]]  # one or many indices

        # Recall@k
        for k in ks:
            topk = ranks[:k].tolist()
            if any(g in topk for g in gold_indices):
                recall_counts[k] += 1

        # MRR: rank of first correct answer
        best_rank = None
        for g in gold_indices:
            pos = (ranks == g).nonzero(as_tuple=False)
            if pos.numel() > 0:
                rank_pos = pos[0].item()
                if best_rank is None or rank_pos < best_rank:
                    best_rank = rank_pos
        if best_rank is not None:
            mrr_sum += 1.0 / (best_rank + 1)

    recall_metrics = {f"recall@{k}": recall_counts[k] / total for k in ks}
    mrr = mrr_sum / total
    return recall_metrics, mrr


def train(model_name, train_path, val_path, out_dir, batch, epochs, lr):
    device = "cuda" 
    model = SentenceTransformer(model_name, device=device)

    # Load train data
    train_data = load_train_data(train_path)
    print(f"Train samples: {len(train_data)}")
    train_loader = DataLoader(train_data, batch_size=batch, shuffle=True)

    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Load validation data
    print("Loading validation data")
    val_data = load_val_data(val_path)
    print(f"Val samples: {len(val_data)}")

    best_mrr = -1.0
    best_dir = os.path.join(out_dir, "best_model")
    os.makedirs(best_dir, exist_ok=True)

    for epoch in range(epochs):
        print(f"===== Epoch {epoch+1}/{epochs} =====")

        # SentenceTransformers handles optimizer & scheduler internally
        model.fit(
            train_objectives=[(train_loader, train_loss)],
            epochs=1,
            # warmup_ratio=0.1,
            scheduler="warmupcosine", 
            optimizer_params={"lr": lr},
            use_amp=True,         
            show_progress_bar=True,
        )

        # Save epoch checkpoint
        epoch_dir = os.path.join(out_dir, f"epoch_{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)
        model.save(epoch_dir)
        print(f"Saved checkpoint: {epoch_dir}")

        # Evaluate retriever on validation set
        recall_metrics, mrr = evaluate_retriever(model, val_data, ks=(5, 10))
        r5 = recall_metrics["recall@5"]
        r10 = recall_metrics["recall@10"]
        print(f"Recall@5 = {r5:.4f} | Recall@10 = {r10:.4f} | MRR = {mrr:.4f}")

        # 6. Track best model by MRR
        if mrr > best_mrr:
            best_mrr = mrr
            model.save(best_dir)
            print(f"[INFO] Best model saved (MRR={best_mrr:.4f})")

    print(f"Best MRR = {best_mrr:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BGE-M3 retriever with val eval")
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--val", type=str, required=True)
    parser.add_argument("--model", type=str, default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--out", type=str, default="bge-base-en-v1.5-finetuned")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    train(
        model_name=args.model,
        train_path=args.train,
        val_path=args.val,
        out_dir=args.out,
        batch=args.batch,
        epochs=args.epochs,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
