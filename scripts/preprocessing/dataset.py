import argparse
import json
from tqdm import tqdm
import pandas as pd
from rank_bm25 import BM25Okapi
from collections import defaultdict
import random
import re

def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = re.sub(r'\s+', ' ', s)
    return s

def load_data(file_path):
    df = pd.read_csv(file_path)
    records = df.to_dict(orient='records')
    data = []
    for item in records:
        # breakpoint()
        q = clean_text(item.get('question', ''))
        a = clean_text(item.get('answer', ''))
        # t = clean_text(item.get('theme', ''))
        if q and a:
            data.append({'question': q, 'answer': a})
    return data

def group_answer(data):
    answer_group = defaultdict(list)
    for item in data:
        answer_group[item['answer']].append(item['question'])
    return answer_group

def build_bm25(corpus):
    tokenized = [doc.split() for doc in corpus]
    return BM25Okapi(tokenized)

def get_negatives(query, pos_answer, corpus, bm25, topn=20, neg_k=5):
    bm25_top = bm25.get_top_n(query.split(), corpus, n=topn)
    hard_negs = [a for a in bm25_top if a != pos_answer]
    
    if len(hard_negs) < neg_k:
        remaining = [x for x in corpus if x != pos_answer and x not in hard_negs]
        hard_negs += random.sample(remaining, neg_k - len(hard_negs))

    return hard_negs[:neg_k]

def create_data(data, neg_k=5):
    answer_group = group_answer(data)
    corpus = list(answer_group.keys())
    bm25 = build_bm25(corpus)

    dataset = []
    for item in tqdm(data, desc="Creating dataset"):
        q = item['question']
        pos = item['answer']
        negs = get_negatives(q, pos, corpus, bm25, neg_k=neg_k)

        dataset.append({
            'question': q,
            'positive': pos,
            'negatives': negs
        })
    return dataset

def save_dataset(dataset, path):
    with open(path, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--neg_k', type=int, default=5)
    args = parser.parse_args()

    print("[INFO] Loading data")
    data = load_data(args.input)

    print("[INFO] Creating dataset with hard negatives")
    dataset = create_data(data, neg_k=args.neg_k)

    print(f"[INFO] Saving dataset to {args.output}")
    save_dataset(dataset, args.output)

    print("[INFO] Done.")

if __name__ == "__main__":
    main()