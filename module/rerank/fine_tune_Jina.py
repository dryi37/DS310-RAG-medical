import argparse
import os
import json
import random
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    set_seed,
)

os.environ["WANDB_DISABLED"] = "true"


# ======================
# Utils
# ======================
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def norm(s: str) -> str:
    return (s or "").strip()


# ======================
# Dataset (group-based)
# ======================
class GroupRerankDataset(Dataset):
    """
    Each item:
    {
        "question": "...",
        "positive": "...",
        "negatives": [...]
    }
    """

    def __init__(
        self,
        items,
        tokenizer,
        max_length=512,
        max_negs=8,
        shuffle_negs=True,
    ):
        self.items = items
        self.tok = tokenizer
        self.max_length = max_length
        self.max_negs = max_negs
        self.shuffle_negs = shuffle_negs

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        q = norm(it.get("question", ""))
        pos = norm(it.get("positive", ""))

        negs = it.get("negatives", []) or []
        negs = [norm(x) for x in negs if norm(x)]

        if self.shuffle_negs:
            random.shuffle(negs)

        negs = negs[: self.max_negs]

        # group = [positive, negatives...]
        cands = [pos] + negs

        enc = self.tok(
            [q] * len(cands),
            cands,
            truncation=True,
            max_length=self.max_length,
        )

        enc["labels"] = 0            # correct index
        enc["group_size"] = len(cands)
        return enc


# ======================
# Collator
# ======================
class GroupCollator:
    def __init__(self, tokenizer):
        self.tok = tokenizer

    def __call__(self, batch):
        group_sizes = [b["group_size"] for b in batch]
        labels = torch.tensor([b["labels"] for b in batch], dtype=torch.long)

        flat = {}
        keys = [k for k in batch[0].keys() if k not in ("labels", "group_size")]

        for k in keys:
            flat[k] = []
            for b in batch:
                flat[k].extend(b[k])

        padded = self.tok.pad(flat, padding=True, return_tensors="pt")
        padded["labels"] = labels
        padded["group_sizes"] = torch.tensor(group_sizes, dtype=torch.long)
        return padded


# ======================
# Trainer (Listwise CE)
# ======================
class GroupRerankTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")            # (B,)
        group_sizes = inputs.pop("group_sizes")  # (B,)

        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)      # (sum_group,)

        splits = torch.split(logits, group_sizes.tolist())

        loss = 0.0
        for i, group_logits in enumerate(splits):
            group_logits = group_logits.unsqueeze(0)  # (1, G)
            loss += torch.nn.functional.cross_entropy(
                group_logits,
                labels[i].unsqueeze(0)
            )

        loss = loss / labels.size(0)
        return (loss, outputs) if return_outputs else loss


# ======================
# Evaluation
# ======================
@torch.no_grad()
def evaluate_mrr_recall_group(
    model,
    tokenizer,
    val_items,
    device,
    max_length=512,
    max_negs=20,
    ks=(5, 10),
):
    model.eval()

    total = 0
    mrr_sum = 0.0
    recall_cnt = {k: 0 for k in ks}

    for it in val_items:
        q = norm(it.get("question", ""))
        pos = norm(it.get("positive", ""))
        negs = it.get("negatives", []) or []
        negs = [norm(x) for x in negs if norm(x)]
        negs = negs[:max_negs]

        cands = [pos] + negs
        if len(cands) < 2:
            continue

        enc = tokenizer(
            [q] * len(cands),
            cands,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        scores = model(**enc).logits.squeeze(-1).float().tolist()
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        gold_rank = ranked.index(0)

        total += 1
        mrr_sum += 1.0 / (gold_rank + 1)

        for k in ks:
            if gold_rank < k:
                recall_cnt[k] += 1

    if total == 0:
        return {f"recall@{k}": 0.0 for k in ks}, 0.0

    recall = {f"recall@{k}": recall_cnt[k] / total for k in ks}
    mrr = mrr_sum / total
    return recall, mrr


# ======================
# Callback
# ======================
class EvalCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        val_items,
        device,
        out_dir,
        max_length=512,
        max_negs=20,
    ):
        self.tokenizer = tokenizer
        self.val_items = val_items
        self.device = device
        self.max_length = max_length
        self.max_negs = max_negs

        self.best_mrr = -1.0
        self.best_dir = os.path.join(out_dir, "best_model")
        os.makedirs(self.best_dir, exist_ok=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs["model"].to(self.device)

        recall, mrr = evaluate_mrr_recall_group(
            model,
            self.tokenizer,
            self.val_items,
            self.device,
            self.max_length,
            self.max_negs,
        )

        print(
            f"\n[VAL] epoch={int(state.epoch)} "
            f"recall@5={recall['recall@5']:.4f} "
            f"recall@10={recall['recall@10']:.4f} "
            f"MRR={mrr:.4f}"
        )

        if mrr > self.best_mrr:
            self.best_mrr = mrr
            model.save_pretrained(self.best_dir)
            self.tokenizer.save_pretrained(self.best_dir)
            print(f"[SAVE] best model (MRR={mrr:.4f})")


# ======================
# Main
# ======================
def main():
    parser = argparse.ArgumentParser("Fine-tune Jina Reranker v2 (listwise CE)")
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument(
        "--reranker",
        default="jinaai/jina-reranker-v2-base-multilingual",
    )
    parser.add_argument("--out", default="jina_reranker_out")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)

    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_negs_train", type=int, default=8)
    parser.add_argument("--max_negs_val", type=int, default=20)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_items = read_jsonl(args.train)
    val_items = read_jsonl(args.val)

    # 🔴 JINA: trust_remote_code REQUIRED
    tokenizer = AutoTokenizer.from_pretrained(
        args.reranker,
        trust_remote_code=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.reranker,
        trust_remote_code=True,
    ).to(device)

    train_ds = GroupRerankDataset(
        train_items,
        tokenizer,
        max_length=args.max_length,
        max_negs=args.max_negs_train,
        shuffle_negs=True,
    )

    collator = GroupCollator(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=50,
        fp16=(args.fp16 and device == "cuda"),
        report_to=[],
        evaluation_strategy="no",
        save_strategy="no",
    )

    trainer = GroupRerankTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.add_callback(
        EvalCallback(
            tokenizer,
            val_items,
            device,
            args.out,
            args.max_length,
            args.max_negs_val,
        )
    )

    trainer.train()
    print("[DONE] Training finished.")


if __name__ == "__main__":
    main()
