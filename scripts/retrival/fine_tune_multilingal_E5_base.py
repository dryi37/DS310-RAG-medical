import argparse
import os
import json
import random
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

os.environ["WANDB_DISABLED"] = "true"

Q_PREFIX = "query: "
P_PREFIX = "passage: "


# -----------------------------
# Train dataset (SAMPLE ONLY)
# -----------------------------
class TripletSampleDataset(Dataset):
    """
    JSONL lines:
      {"question": q, "positive": pos, "negatives": [neg1, neg2, ...]}

    Each __getitem__ samples 1 negative from negatives list.
    Returns InputExample(texts=[query:q, passage:pos, passage:neg])
    """
    def __init__(self, path: str, seed: int = 42, min_negs: int = 1):
        self.items = []
        self.rng = random.Random(seed)

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                it = json.loads(line)
                q = (it.get("question") or "").strip()
                pos = (it.get("positive") or "").strip()
                negs = it.get("negatives") or []

                negs = [n.strip() for n in negs if isinstance(n, str) and n.strip()]
                # remove duplicates + remove pos
                negs = list(dict.fromkeys([n for n in negs if n != pos]))

                if q and pos and len(negs) >= min_negs:
                    self.items.append({"q": q, "pos": pos, "negs": negs})

        if not self.items:
            raise ValueError("No valid training samples. Need: question, positive, negatives[]")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        neg = self.rng.choice(it["negs"])
        return InputExample(texts=[Q_PREFIX + it["q"], P_PREFIX + it["pos"], P_PREFIX + neg])


# -----------------------------
# Validation data + eval
# -----------------------------
def load_val_data(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            it = json.loads(line)
            q = (it.get("question") or "").strip()
            pos = (it.get("positive") or "").strip()
            if q and pos:
                data.append({"question": q, "positive": pos})
    return data


def evaluate_retriever(model, val_data, ks=(1, 5, 10), batch_size=32):
    model.eval()

    questions_raw = [it["question"] for it in val_data]
    answers_raw = [it["positive"] for it in val_data]

    ans_to_indices = defaultdict(list)
    for idx, ans in enumerate(answers_raw):
        ans_to_indices[ans].append(idx)

    # E5 prefixes
    questions = [Q_PREFIX + q for q in questions_raw]
    answers = [P_PREFIX + a for a in answers_raw]

    with torch.no_grad():
        q_emb = model.encode(questions, convert_to_tensor=True, batch_size=batch_size)
        a_emb = model.encode(answers, convert_to_tensor=True, batch_size=batch_size)

    q_emb = F.normalize(q_emb, p=2, dim=1)
    a_emb = F.normalize(a_emb, p=2, dim=1)

    sims = torch.matmul(q_emb, a_emb.T)

    total = sims.size(0)
    recall_counts = {k: 0 for k in ks}
    mrr_sum = 0.0

    for i in range(total):
        ranks = torch.argsort(sims[i], descending=True)
        gold_indices = ans_to_indices[answers_raw[i]]

        for k in ks:
            topk = ranks[:k].tolist()
            if any(g in topk for g in gold_indices):
                recall_counts[k] += 1

        best_rank = None
        for g in gold_indices:
            pos = (ranks == g).nonzero(as_tuple=False)
            if pos.numel() > 0:
                r = pos[0].item()
                if best_rank is None or r < best_rank:
                    best_rank = r
        if best_rank is not None:
            mrr_sum += 1.0 / (best_rank + 1)

    recall_metrics = {f"recall@{k}": recall_counts[k] / total for k in ks}
    mrr = mrr_sum / total
    return recall_metrics, mrr


# -----------------------------
# Train loop
# -----------------------------
def train(model_name, train_path, val_path, out_dir, batch, epochs, lr, seed):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)

    train_dataset = TripletSampleDataset(train_path, seed=seed, min_negs=1)
    print(f"Train samples: {len(train_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, drop_last=True)

    train_loss = losses.TripletLoss(model)

    print("Loading validation data")
    val_data = load_val_data(val_path)
    print(f"Val samples: {len(val_data)}")

    best_mrr = -1.0
    best_dir = os.path.join(out_dir, "best_model")
    os.makedirs(best_dir, exist_ok=True)

    for epoch in range(epochs):
        print(f"===== Epoch {epoch+1}/{epochs} =====")

        model.fit(
            train_objectives=[(train_loader, train_loss)],
            epochs=1,
            scheduler="warmupcosine",
            optimizer_params={"lr": lr},
            use_amp=(device == "cuda"),
            show_progress_bar=True,
        )

        epoch_dir = os.path.join(out_dir, f"epoch_{epoch+1}")
        os.makedirs(epoch_dir, exist_ok=True)
        model.save(epoch_dir)
        print(f"Saved checkpoint: {epoch_dir}")

        recall_metrics, mrr = evaluate_retriever(model, val_data, ks=(1, 5, 10))
        print(
            f"Recall@1 = {recall_metrics['recall@1']:.4f} | "
            f"Recall@5 = {recall_metrics['recall@5']:.4f} | "
            f"Recall@10 = {recall_metrics['recall@10']:.4f} | "
            f"MRR = {mrr:.4f}"
        )

        if mrr > best_mrr:
            best_mrr = mrr
            model.save(best_dir)
            print(f"[INFO] Best model saved (MRR={best_mrr:.4f})")

    print(f"Best MRR = {best_mrr:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune multilingual-e5-base with TripletLoss (sample negatives)")
    parser.add_argument("--train", type=str, required=True, help="train JSONL with question/positive/negatives")
    parser.add_argument("--val", type=str, required=True, help="val JSONL with question/positive")
    parser.add_argument("--model", type=str, default="intfloat/multilingual-e5-small")
    parser.add_argument("--out", type=str, default="multilingual-e5-base-triplet-sample")
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)

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
        seed=args.seed,
    )


if __name__ == "__main__":
    main()