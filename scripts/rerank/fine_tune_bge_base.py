import argparse
import os
import json
import random
from typing import List, Dict, Any, Tuple

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)
from sentence_transformers import SentenceTransformer

os.environ["WANDB_DISABLED"] = "true"


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def norm(s: str) -> str:
    return (s or "").strip()


# -------------------------
# Hard negative mining (IGNORE dataset negatives)
# -------------------------
@torch.no_grad()
def mine_hard_negatives_ignore_field(
    items: List[Dict[str, Any]],
    retriever_name: str = "intfloat/e5-small-v2",
    neg_per_pos: int = 4,
    mine_top: int = 50,          # search depth (top candidates to consider)
    pool_max: int = 200000,
    encode_bs: int = 64,
    device: str = "cuda",
    seed: int = 42,
) -> List[List[str]]:
    """
    Mine negatives ONLY from other items' POSITIVES.
    Completely ignores any `negatives` field in your data.
    """
    rng = random.Random(seed)

    # Pool = positives from items (optionally capped)
    positives = [norm(it.get("positive", "")) for it in items]
    positives = [p for p in positives if p]

    if len(positives) > pool_max:
        idxs = list(range(len(positives)))
        rng.shuffle(idxs)
        idxs = idxs[:pool_max]
        positives_pool = [positives[j] for j in idxs]
    else:
        positives_pool = positives

    item_pos = [norm(it.get("positive", "")) for it in items]

    retriever = SentenceTransformer(
        retriever_name,
        device=device
    )

    # E5 prefix
    queries = ["query: " + norm(it.get("question", "")) for it in items]
    passages = ["passage: " + p for p in positives_pool]

    q_emb = retriever.encode(queries, convert_to_tensor=True, batch_size=encode_bs)
    p_emb = retriever.encode(passages, convert_to_tensor=True, batch_size=encode_bs)

    q_emb = F.normalize(q_emb, p=2, dim=1)
    p_emb = F.normalize(p_emb, p=2, dim=1)

    sims = torch.matmul(q_emb, p_emb.T)  # (num_items, pool)

    negs_per_item: List[List[str]] = []
    for i in range(len(items)):
        q = norm(items[i].get("question", ""))
        pos = item_pos[i]
        if not q or not pos:
            negs_per_item.append([])
            continue

        ranked = torch.argsort(sims[i], descending=True).tolist()

        chosen = []
        # look into top `mine_top` first (hard-ish). If not enough, continue scanning.
        scan_list = ranked[:max(mine_top, neg_per_pos * 5)]  # safety depth
        for j in scan_list:
            cand = positives_pool[j]
            if cand and cand != pos:
                chosen.append(cand)
            if len(chosen) >= neg_per_pos:
                break

        # If still not enough (rare), fallback deeper
        if len(chosen) < neg_per_pos:
            for j in ranked[len(scan_list):]:
                cand = positives_pool[j]
                if cand and cand != pos:
                    chosen.append(cand)
                if len(chosen) >= neg_per_pos:
                    break

        negs_per_item.append(chosen)

    return negs_per_item


# -------------------------
# Dataset: (q, passage) -> label
# -------------------------
class RerankBinaryDataset(Dataset):
    def __init__(
        self,
        items: List[Dict[str, Any]],
        mined_negs: List[List[str]],
        tokenizer,
        max_length: int = 256,
    ):
        self.tok = tokenizer
        self.max_length = max_length
        self.examples: List[Tuple[str, str, float]] = []

        for it, negs in zip(items, mined_negs):
            q = norm(it.get("question", ""))
            pos = norm(it.get("positive", ""))
            if not q or not pos:
                continue

            self.examples.append((q, pos, 1.0))
            for neg in negs:
                if neg:
                    self.examples.append((q, neg, 0.0))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        q, p, y = self.examples[idx]
        enc = self.tok(q, p, truncation=True, max_length=self.max_length)
        enc["labels"] = float(y)
        return enc


class RerankTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)
        loss = torch.nn.BCEWithLogitsLoss()(logits, labels)
        return (loss, outputs) if return_outputs else loss


@torch.no_grad()
def evaluate_mrr_recall(
    model,
    tokenizer,
    val_items: List[Dict[str, Any]],
    val_negs: List[List[str]],
    device: str,
    max_length: int = 256,
    ks=(5, 10),
):
    model.eval()
    total = 0
    recall_counts = {k: 0 for k in ks}
    mrr_sum = 0.0

    for it, negs in zip(val_items, val_negs):
        q = norm(it.get("question", ""))
        pos = norm(it.get("positive", ""))
        if not q or not pos:
            continue

        cands = [pos] + [n for n in negs if n]
        if len(cands) < 2:
            continue

        scores = []
        bs = 32
        for start in range(0, len(cands), bs):
            chunk = cands[start : start + bs]
            enc = tokenizer(
                [q] * len(chunk),
                chunk,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            out = model(**enc)
            scores.extend(out.logits.squeeze(-1).float().cpu().tolist())

        ranked = sorted(range(len(cands)), key=lambda i: scores[i], reverse=True)
        gold_rank = ranked.index(0)
        total += 1

        for k in ks:
            if gold_rank < k:
                recall_counts[k] += 1
        mrr_sum += 1.0 / (gold_rank + 1)

    if total == 0:
        return {f"recall@{k}": 0.0 for k in ks}, 0.0

    recall = {f"recall@{k}": recall_counts[k] / total for k in ks}
    mrr = mrr_sum / total
    return recall, mrr


def main():
    parser = argparse.ArgumentParser("Fine-tune reranker (IGNORE provided negatives, auto-mine)")
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--val", type=str, required=True)

    parser.add_argument("--reranker", type=str, default="BAAI/bge-reranker-base")
    parser.add_argument("--retriever", type=str, required=True)
    parser.add_argument("--out", type=str, default="reranker_out")

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=1)

    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--neg_per_pos", type=int, default=4)
    parser.add_argument("--mine_top", type=int, default=50)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_items = read_jsonl(args.train)
    val_items = read_jsonl(args.val)

    print("[INFO] Mining hard negatives for TRAIN (ignoring dataset negatives)...")
    train_negs = mine_hard_negatives_ignore_field(
        train_items,
        retriever_name=args.retriever,
        neg_per_pos=args.neg_per_pos,
        mine_top=args.mine_top,
        device=device,
        seed=args.seed,
    )

    print("[INFO] Mining hard negatives for VAL (ignoring dataset negatives)...")
    val_negs = mine_hard_negatives_ignore_field(
        val_items,
        retriever_name=args.retriever,
        neg_per_pos=max(10, args.neg_per_pos),
        mine_top=max(args.mine_top, 100),
        device=device,
        seed=args.seed + 999,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.reranker, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.reranker, num_labels=1).to(device)

    train_ds = RerankBinaryDataset(train_items, train_negs, tokenizer, max_length=args.max_length)
    val_ds = RerankBinaryDataset(val_items, val_negs, tokenizer, max_length=args.max_length)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=50,              # sẽ log loss
        logging_strategy="steps",
        fp16=(args.fp16 and device == "cuda"),
        report_to=[],
    )

    trainer = RerankTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,  # eval loss (tham khảo)
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    best_mrr = -1.0
    best_dir = os.path.join(args.out, "best_model")
    os.makedirs(best_dir, exist_ok=True)

    for ep in range(args.epochs):
        trainer.train()

        recall, mrr = evaluate_mrr_recall(
            model=trainer.model,
            tokenizer=tokenizer,
            val_items=val_items,
            val_negs=val_negs,
            device=device,
            max_length=args.max_length,
            ks=(5, 10),
        )
        print(
            f"[VAL] epoch={ep+1} "
            f"recall@5={recall['recall@5']:.4f} "
            f"recall@10={recall['recall@10']:.4f} "
            f"MRR={mrr:.4f}"
        )

        if mrr > best_mrr:
            best_mrr = mrr
            trainer.model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"[INFO] Saved best_model to {best_dir} (MRR={best_mrr:.4f})")

    print(f"Best MRR = {best_mrr:.4f}")


if __name__ == "__main__":
    main()