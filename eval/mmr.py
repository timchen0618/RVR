import json
import sys
from typing import List, Dict, Any
import numpy as np


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def score_to_vector(doc: Dict[str, Any], all_docs: List[Dict[str, Any]]) -> np.ndarray:
    """
    Since we don't have dense embeddings, we represent each document
    as a 1D vector using its normalized retrieval score rank position.
    For sim(d_i, q): use the original retrieval score directly.
    For sim(d_i, d_j): use overlap of top-k BM25-style score buckets.
    
    In practice without embeddings, we approximate:
      - sim(d_i, q)    = normalized retrieval score (position in ranked list)
      - sim(d_i, d_j)  = 1 - |rank_i - rank_j| / N  (rank proximity as diversity proxy)
    """
    pass  # Not used directly — see mmr() below


def mmr(
    docs: List[Dict[str, Any]],
    top_k: int = 100,
    lambda_param: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Apply Maximal Marginal Relevance to select top_k documents from docs.

    MMR = argmax_{d_i in C\S} [ lambda * sim(d_i, q) - (1 - lambda) * max_{d_j in S} sim(d_i, d_j) ]

    Since we don't have dense embeddings, we use:
      - sim(d_i, q):    normalized retrieval score (float score field, or rank-based fallback)
      - sim(d_i, d_j):  TF-style token overlap (Jaccard) on document text as diversity proxy

    Args:
        docs:         Ranked list of candidate documents (up to 200)
        top_k:        Number of documents to select (default 100)
        lambda_param: Trade-off between relevance and diversity (default 0.5)

    Returns:
        Selected documents in MMR order
    """
    if not docs:
        return []

    top_k = min(top_k, len(docs))
    n = len(docs)

    # --- Relevance scores: sim(d_i, q) ---
    # Use the 'score' field if available, otherwise fall back to rank-based score
    raw_scores = []
    for doc in docs:
        s = doc.get("score", None)
        try:
            raw_scores.append(float(s))
        except (TypeError, ValueError):
            raw_scores.append(None)

    if any(s is not None for s in raw_scores):
        # Normalize to [0, 1]
        valid = [s for s in raw_scores if s is not None]
        s_min, s_max = min(valid), max(valid)
        denom = s_max - s_min if s_max != s_min else 1.0
        rel_scores = [
            (s - s_min) / denom if s is not None else 0.0
            for s in raw_scores
        ]
    else:
        # Fallback: rank-based (position 0 = most relevant)
        rel_scores = [1.0 - i / n for i in range(n)]

    # --- Diversity: precompute token sets for Jaccard similarity ---
    def tokenize(text: str):
        return set(text.lower().split())

    token_sets = [tokenize(doc.get("text", "") + " " + doc.get("title", "")) for doc in docs]

    def jaccard(i: int, j: int) -> float:
        a, b = token_sets[i], token_sets[j]
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union > 0 else 0.0

    # --- MMR greedy selection ---
    selected_indices = []
    candidate_indices = list(range(n))

    for _ in range(top_k):
        best_idx = None
        best_score = float("-inf")

        for i in candidate_indices:
            relevance = rel_scores[i]

            if not selected_indices:
                redundancy = 0.0
            else:
                redundancy = max(jaccard(i, j) for j in selected_indices)

            mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        if best_idx is None:
            break

        selected_indices.append(best_idx)
        candidate_indices.remove(best_idx)

    return [dict(docs[i], mmr_score=round(rel_scores[i], 6)) for i in selected_indices]


def apply_mmr_to_file(
    input_file: str,
    output_file: str,
    top_k: int = 100,
    lambda_param: float = 0.5
):
    """
    Read a JSONL file where each entry has up to 200 docs in 'ctxs',
    apply MMR to select top_k, and write output in the same format.
    """
    all_entries = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            all_entries.append(json.loads(line))

    print(f"Loaded {len(all_entries)} entries", flush=True)

    total_before = 0
    total_after = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for idx, entry in enumerate(all_entries):
            ctxs = entry.get("ctxs", [])
            total_before += len(ctxs)

            selected = mmr(ctxs, top_k=top_k, lambda_param=lambda_param)
            total_after += len(selected)

            new_entry = dict(entry)
            new_entry["ctxs"] = selected
            f.write(json.dumps(new_entry, ensure_ascii=False) + "\n")

            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(all_entries)} entries", flush=True)

    avg_before = total_before / len(all_entries) if all_entries else 0
    avg_after = total_after / len(all_entries) if all_entries else 0

    print(f"\n=== MMR Complete ===")
    print(f"Total entries          : {len(all_entries)}")
    print(f"Avg ctxs before MMR    : {avg_before:.1f}")
    print(f"Avg ctxs after MMR     : {avg_after:.1f}")
    print(f"Lambda                 : {lambda_param}")
    print(f"Top-k selected         : {top_k}")
    print(f"Output written to      : {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python mmr_rerank.py <input.jsonl> <output.jsonl> [top_k] [lambda]")
        sys.exit(1)

    input_file  = sys.argv[1]
    output_file = sys.argv[2]
    top_k       = int(sys.argv[3])   if len(sys.argv) > 3 else 100
    lambda_val  = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5

    print(f"Input : {input_file}")
    print(f"Output: {output_file}")
    print(f"Top-k : {top_k}, Lambda: {lambda_val}")

    apply_mmr_to_file(input_file, output_file, top_k=top_k, lambda_param=lambda_val)