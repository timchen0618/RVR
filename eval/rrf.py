import json
import sys
from typing import List, Dict, Any


def reciprocal_rank_fusion(
    lists: List[List[Dict[str, Any]]],
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Perform Reciprocal Rank Fusion over multiple ranked lists of documents.
    Each document is identified by its 'id' field.
    
    Args:
        lists: List of ranked document lists (each from a different retrieval run)
        k: RRF constant (default 60, as in the original paper)
    
    Returns:
        Merged and reranked list of documents, sorted by RRF score descending
    """
    rrf_scores = {}
    doc_store = {}  # Store full doc objects by id

    for ranked_list in lists:
        for rank, doc in enumerate(ranked_list, start=1):
            doc_id = doc.get("id")
            if doc_id is None:
                continue

            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

            # Keep the doc with the highest original score if seen multiple times
            if doc_id not in doc_store:
                doc_store[doc_id] = doc

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    reranked = []
    for doc_id in sorted_ids:
        doc = dict(doc_store[doc_id])
        doc["rrf_score"] = round(rrf_scores[doc_id], 6)
        reranked.append(doc)

    return reranked


def merge_files_with_rrf(
    file1: str,
    file2: str,
    output_file: str,
    k: int = 60
):
    """
    Merge two JSONL files of retrieval results using RRF.
    Entries are matched by 'id' field at the query level.
    Output preserves the input format with updated ctxs.
    """
    def load_jsonl(path: str) -> Dict[str, Dict]:
        entries = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                entries[entry["id"]] = entry
        return entries

    print(f"Loading {file1}...", flush=True)
    entries1 = load_jsonl(file1)
    print(f"Loading {file2}...", flush=True)
    entries2 = load_jsonl(file2)

    all_ids = set(entries1.keys()) | set(entries2.keys())
    print(f"Total unique query ids: {len(all_ids)}", flush=True)
    print(f"Queries in file1 only: {len(set(entries1) - set(entries2))}", flush=True)
    print(f"Queries in file2 only: {len(set(entries2) - set(entries1))}", flush=True)
    print(f"Queries in both files: {len(set(entries1) & set(entries2))}", flush=True)

    merged_entries = []

    for qid in sorted(all_ids):
        entry1 = entries1.get(qid)
        entry2 = entries2.get(qid)

        # Use whichever entry exists as the base (prefer file1)
        base_entry = dict(entry1 if entry1 is not None else entry2)

        # Gather ctxs from each file (empty list if not present)
        ctxs1 = entry1.get("ctxs", []) if entry1 else []
        ctxs2 = entry2.get("ctxs", []) if entry2 else []

        # Run RRF over the two ranked lists
        merged_ctxs = reciprocal_rank_fusion([ctxs1, ctxs2], k=k)

        base_entry["ctxs"] = merged_ctxs
        merged_entries.append(base_entry)

    # Write output as JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in merged_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Stats
    ctx_lengths = [len(e["ctxs"]) for e in merged_entries]
    avg_ctxs = sum(ctx_lengths) / len(ctx_lengths) if ctx_lengths else 0

    print(f"\n=== RRF Merge Complete ===")
    print(f"Total entries written  : {len(merged_entries)}")
    print(f"Avg ctxs per entry     : {avg_ctxs:.1f}")
    print(f"Max ctxs per entry     : {max(ctx_lengths) if ctx_lengths else 0}")
    print(f"Output written to      : {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python rrf_merge.py <file1.jsonl> <file2.jsonl> <output.jsonl> [k]")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]
    output = sys.argv[3]
    k = int(sys.argv[4]) if len(sys.argv) > 4 else 60

    print(f"RRF constant k={k}")
    merge_files_with_rrf(file1, file2, output, k=k)