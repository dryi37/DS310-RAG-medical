import argparse
import json
import random
import os


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def split_dataset(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    random.shuffle(data)
    total = len(data)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_set = data[:train_end]
    val_set = data[train_end:val_end]
    test_set = data[val_end:]

    return train_set, val_set, test_set


def main():
    parser = argparse.ArgumentParser(description="Split pos-neg dataset into train/val/test")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL pos-neg dataset")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory")
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    args = parser.parse_args()

    print("[1] Loading dataset")
    data = load_jsonl(args.input)
    print(f"[INFO] Loaded {len(data)} samples")

    print("[2] Splitting dataset")
    train_set, val_set, test_set = split_dataset(data, args.train, args.val, args.test)

    print("[3] Saving files")
    os.makedirs(args.outdir, exist_ok=True)
    save_jsonl(train_set, f"{args.outdir}/train.jsonl")
    save_jsonl(val_set, f"{args.outdir}/val.jsonl")
    save_jsonl(test_set, f"{args.outdir}/test.jsonl")

    print("[DONE] Dataset successfully split!")
    print(f"Train: {len(train_set)} samples")
    print(f"Val:   {len(val_set)} samples")
    print(f"Test:  {len(test_set)} samples")


if __name__ == "__main__":
    main()