import sys, os

"""
Part 1: Classical Retrieval Methods
- BM25 (sparse retrieval)
- msmarco-MiniLM (dense retrieval)
- UAE-large-v1 (dense retrieval)
"""
import json
import numpy as np
from tqdm import tqdm
import random

# ── BM25 ──────────────────────────────────────────────────────────────────────
from rank_bm25 import BM25Okapi

# ── Dense ─────────────────────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer
import torch


# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    with open("data/test_queries.json") as f:
        test_queries = json.load(f)
    with open("data/train_queries.json") as f:
        train_queries = json.load(f)
    with open("data/tools.json") as f:
        tools = json.load(f)
    return train_queries, test_queries, tools


def recall_at_k(rankings, gold_ids, k):
    """Fraction of queries where gold appears in top-k ranked results."""
    hits = sum(1 for rank_list, gold in zip(rankings, gold_ids)
               if gold in rank_list[:k])
    return hits / len(gold_ids)


# ─────────────────────────────────────────────────────────────────────────────
# BM25
# ─────────────────────────────────────────────────────────────────────────────
def run_bm25(test_queries, tools):
    tool_names = list(tools.keys())
    tool_descs = [tools[n] for n in tool_names]

    # tokenise corpus (tool_id + description)
    corpus_docs = [f"{name} {desc}".lower().split()
                   for name, desc in zip(tool_names, tool_descs)]
    bm25 = BM25Okapi(corpus_docs)

    rankings, gold_ids = [], []
    for sample in tqdm(test_queries, desc="BM25"):
        query_tokens = sample["text"].lower().split()
        scores = bm25.get_scores(query_tokens)
        ranked = np.argsort(scores)[::-1].tolist()
        gold_idx = tool_names.index(sample["gold_tool_name"])
        rankings.append(ranked)
        gold_ids.append(gold_idx)

    r1 = recall_at_k(rankings, gold_ids, 1)
    r5 = recall_at_k(rankings, gold_ids, 5)
    return r1, r5


# ─────────────────────────────────────────────────────────────────────────────
# Dense retrieval
# ─────────────────────────────────────────────────────────────────────────────
def run_dense(test_queries, tools, model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)

    tool_names = list(tools.keys())
    # corpus: "tool_id: <name>\ntool description: <desc>"
    corpus = [f"tool_id: {n}\ntool description: {tools[n]}" for n in tool_names]
    corpus_embs = model.encode(corpus, batch_size=64,
                               convert_to_tensor=True,
                               show_progress_bar=True,
                               normalize_embeddings=True)

    queries_text = [s["text"] for s in test_queries]
    query_embs = model.encode(queries_text, batch_size=64,
                              convert_to_tensor=True,
                              show_progress_bar=True,
                              normalize_embeddings=True)

    # cosine similarity (already L2-normalised → dot product)
    sim = torch.mm(query_embs, corpus_embs.T)  # [Q, D]

    rankings, gold_ids = [], []
    for i, sample in enumerate(test_queries):
        ranked = torch.argsort(sim[i], descending=True).cpu().tolist()
        gold_idx = tool_names.index(sample["gold_tool_name"])
        rankings.append(ranked)
        gold_ids.append(gold_idx)

    r1 = recall_at_k(rankings, gold_ids, 1)
    r5 = recall_at_k(rankings, gold_ids, 5)
    return r1, r5


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_queries, test_queries, tools = load_data()

    print("=" * 60)
    print("Part 1 – Classical Retrieval on test_queries")
    print("=" * 60)

    # ── BM25 ──────────────────────────────────────────────────────────────────
    print("\n[1/3] BM25 …")
    r1_bm25, r5_bm25 = run_bm25(test_queries, tools)
    print(f"  BM25          Recall@1={r1_bm25:.4f}  Recall@5={r5_bm25:.4f}")

    # ── MiniLM (msmarco) ──────────────────────────────────────────────────────
    print("\n[2/3] msmarco-MiniLM …")
    r1_mini, r5_mini = run_dense(
        test_queries, tools,
        "sentence-transformers/msmarco-MiniLM-L6-cos-v5"
        # fallback bi-encoder:  "sentence-transformers/msmarco-MiniLM-L6-cos-v5"
    )
    print(f"  msmarco-MiniLM  Recall@1={r1_mini:.4f}  Recall@5={r5_mini:.4f}")

    # ── UAE-Large-V1 ──────────────────────────────────────────────────────────
    print("\n[3/3] UAE-Large-V1 …")
    r1_uae, r5_uae = run_dense(
        test_queries, tools,
        "WhereIsAI/UAE-Large-V1"
    )
    print(f"  UAE-Large-V1    Recall@1={r1_uae:.4f}  Recall@5={r5_uae:.4f}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"{'Method':<20} {'Recall@1':>10} {'Recall@5':>10}")
    print("-" * 42)
    print(f"{'BM25':<20} {r1_bm25:>10.4f} {r5_bm25:>10.4f}")
    print(f"{'msmarco-MiniLM':<20} {r1_mini:>10.4f} {r5_mini:>10.4f}")
    print(f"{'UAE-Large-V1':<20} {r1_uae:>10.4f} {r5_uae:>10.4f}")
    print("=" * 60)
