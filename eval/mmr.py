import json
import sys
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer


def mmr(
    query: str,
    docs: List[Dict[str, Any]],
    model: SentenceTransformer,
    top_k: int = 100,
    lambda_param: float = 0.5,
    batch_size: int = 64
) -> List[Dict[str, Any]]:
    """
    Apply Maximal Marginal Relevance to select top_k documents from docs.

    MMR = argmax_{d_i in C\S} [ lambda * sim(d_i, q) - (1 - lambda) * max_{d_j in S} sim(d_i, d_j) ]

    sim(d_i, q) and sim(d_i, d_j) are cosine similarities over sentence-transformer embeddings.

    Args:
        query:        The query string
        docs:         Candidate documents (up to 200), each with 'text' and 'title' fields
        model:        SentenceTransformer encoder
        top_k:        Number of documents to select (default 100)
        lambda_param: Trade-off between relevance and diversity (default 0.5)
        batch_size:   Encoding batch size

    Returns:
        Selected documents in MMR order with mmr_score attached
    """
    if not docs:
        return []

    top_k = min(top_k, len(docs))
    n = len(docs)

    # Build text to encode: title + text for each doc
    doc_texts = [
        (doc.get("title", "") + " " + doc.get("text", "")).strip()
        for doc in docs
    ]

    # Encode query and all documents in one batched call
    all_texts = [query] + doc_texts
    embeddings = model.encode(
        all_texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True  # L2-normalize so dot product = cosine similarity
    )

    query_emb = embeddings[0:1]   # (1, d)
    doc_embs  = embeddings[1:]    # (n, d)

    # sim(d_i, q) for all docs: shape (n,)
    sim_to_query = (doc_embs @ query_emb.T).squeeze(1)

    # Precompute full doc-doc cosine similarity matrix: shape (n, n)
    sim_doc_doc = doc_embs @ doc_embs.T

    # --- MMR greedy selection ---
    selected_indices = []
    candidate_indices = list(range(n))

    for _ in range(top_k):
        best_idx = None
        best_score = float("-inf")

        for i in candidate_indices:
            relevance = float(sim_to_query[i])

            if not selected_indices:
                redundancy = 0.0
            else:
                redundancy = float(np.max(sim_doc_doc[i, selected_indices]))

            mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        if best_idx is None:
            break

        selected_indices.append(best_idx)
        candidate_indices.remove(best_idx)

    return [
        dict(docs[i], mmr_score=round(float(sim_to_query[i]), 6))
        for i in selected_indices
    ]


def apply_mmr_to_file(
    input_file: str,
    output_file: str,
    top_k: int = 100,
    lambda_param: float = 0.5,
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64
):
    """
    Read a JSONL file where each entry has up to 200 docs in 'ctxs',
    apply MMR to select top_k, and write output in the same format.
    """
    # Load encoder once for all queries
    print(f"Loading encoder: {encoder_name}", flush=True)
    model = SentenceTransformer(encoder_name)
    print("Encoder loaded.", flush=True)

    all_entries = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            all_entries.append(json.loads(line))

    print(f"Loaded {len(all_entries)} entries from {input_file}", flush=True)

    total_before = 0
    total_after  = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for idx, entry in enumerate(all_entries):
            query = entry.get("question", entry.get("input", ""))
            ctxs  = entry.get("ctxs", [])
            total_before += len(ctxs)

            selected = mmr(
                query=query,
                docs=ctxs,
                model=model,
                top_k=top_k,
                lambda_param=lambda_param,
                batch_size=batch_size
            )
            total_after += len(selected)

            new_entry = dict(entry)
            new_entry["ctxs"] = selected
            f.write(json.dumps(new_entry, ensure_ascii=False) + "\n")

            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(all_entries)} entries", flush=True)

    avg_before = total_before / len(all_entries) if all_entries else 0
    avg_after  = total_after  / len(all_entries) if all_entries else 0

    print(f"\n=== MMR Complete ===")
    print(f"Encoder                : {encoder_name}")
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

    print(f"Input    : {input_file}")
    print(f"Output   : {output_file}")
    print(f"Top-k    : {top_k}")
    print(f"Lambda   : {lambda_val}")
# /scratch/dq2024/diverse_retriever/inference/contriever/base/qampari_corpus_finetuned_base_sample_0_t200.json
# /scratch/dq2024/diverse_retriever/inference/infly/base/qampari_base_finetuned_infly_inp0_t200.json

    apply_mmr_to_file(
        input_file=input_file,
        output_file=output_file,
        top_k=top_k,
        lambda_param=lambda_val
    )