# audit_query_vectors.py
import numpy as np, hashlib, os, itertools as it

files = [
        '/scratch/dq2024/diverse_retriever/training/contriever/eval/gru/two_stage/query_embeddings_qampari_corpus_only_random_gold_AR_r1contr_r2gru_100.json.npy',
        '/scratch/dq2024/diverse_retriever/training/contriever/eval/gru/two_stage/query_embeddings_qampari_corpus_only_random_gold_AR_r1contr_r2gru_0123_100.json.npy',
        '/scratch/dq2024/diverse_retriever/training/contriever/eval/standard/query_embeddings_qampari_corpus_only_random_gold_AR_r1contr_r2std_100.json.npy',
        '/scratch/dq2024/diverse_retriever/training/contriever/eval/standard/query_embeddings_qampari_corpus_only_random_gold_AR_r1contr_r2std_0123_100.json.npy',

        '/scratch/dq2024/diverse_retriever/training/contriever/eval/gru/two_stage/query_embeddings_qampari_corpus_only_random_gold_ER_r1contr_r2gru_100.json.npy',
        '/scratch/dq2024/diverse_retriever/training/contriever/eval/gru/two_stage/query_embeddings_qampari_corpus_only_random_gold_ER_r1contr_r2gru_0123_100.json.npy',
        '/scratch/dq2024/diverse_retriever/training/contriever/eval/standard/query_embeddings_qampari_corpus_only_random_gold_ER_r1contr_r2std_100.json.npy',
        '/scratch/dq2024/diverse_retriever/training/contriever/eval/standard/query_embeddings_qampari_corpus_only_random_gold_ER_r1contr_r2std_0123_100.json.npy',

        '/scratch/dq2024/diverse_retriever/training/contriever/eval/gru/two_stage/query_embeddings_qampari_corpus_only_random_gold_AR_r1contr_r2gru_200.json.npy',
        '/scratch/dq2024/diverse_retriever/training/contriever/eval/gru/two_stage/query_embeddings_qampari_corpus_only_random_gold_AR_r1contr_r2gru_0123_200.json.npy',
        '/scratch/dq2024/diverse_retriever/training/contriever/eval/standard/query_embeddings_qampari_corpus_only_random_gold_AR_r1contr_r2std_200.json.npy',
        '/scratch/dq2024/diverse_retriever/training/contriever/eval/standard/query_embeddings_qampari_corpus_only_random_gold_AR_r1contr_r2std_0123_200.json.npy',

        '/scratch/dq2024/diverse_retriever/training/contriever/eval/gru/two_stage/query_embeddings_qampari_corpus_only_random_gold_ER_r1contr_r2gru_200.json.npy',
        '/scratch/dq2024/diverse_retriever/training/contriever/eval/gru/two_stage/query_embeddings_qampari_corpus_only_random_gold_ER_r1contr_r2gru_0123_200.json.npy',
        '/scratch/dq2024/diverse_retriever/training/contriever/eval/standard/query_embeddings_qampari_corpus_only_random_gold_ER_r1contr_r2std_200.json.npy',
        '/scratch/dq2024/diverse_retriever/training/contriever/eval/standard/query_embeddings_qampari_corpus_only_random_gold_ER_r1contr_r2std_0123_200.json.npy',
        
    ]

def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

# 1) File-level hashes & basic stats
meta = []
arrays = []
for f in files:
    x = np.load(f)
    arrays.append(x)
    meta.append({
        "file": f,
        "sha256": sha256(f),
        "shape": x.shape,
        "dtype": str(x.dtype),
        "row_l2_mean": float(np.linalg.norm(x, axis=1).mean()),
        "row_l2_std": float(np.linalg.norm(x, axis=1).std()),
        "global_mean": float(x.mean()),
        "global_std": float(x.std()),
    })

print("=== File hashes & stats ===")
for m in meta:
    print(os.path.basename(m["file"]), m["sha256"][:16], m["shape"], m["dtype"],
          f"rowL2 μ={m['row_l2_mean']:.6f} σ={m['row_l2_std']:.6f}",
          f"glob μ={m['global_mean']:.6f} σ={m['global_std']:.6f}")

# 2) Exact/near-exact duplicates
print("\n=== Exact/near-duplicate check (max |Δ|) ===")
for (i,fi),(j,gj) in it.combinations(enumerate(files), 2):
    if arrays[i].shape != arrays[j].shape: continue
    max_abs = float(np.max(np.abs(arrays[i]-arrays[j])))
    if max_abs == 0.0 or max_abs < 1e-12:
        print(f"IDENTICAL: {os.path.basename(fi)} == {os.path.basename(gj)} (max|Δ|={max_abs:.1e})")

# 3) Are rows aligned to identical queries?
# If they are, cosine should be meaningful; if not, it’s garbage-in.
def cosine_rowwise(A,B):
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    return (A*B).sum(axis=1)

print("\n=== Pairwise avg cosine between ALL query files ===")
for (i,fi),(j,gj) in it.combinations(enumerate(files), 2):
    if arrays[i].shape != arrays[j].shape: continue
    cs = cosine_rowwise(arrays[i], arrays[j])
    print(f"{os.path.basename(fi)}  vs  {os.path.basename(gj)} :  mean={cs.mean():.4f}  std={cs.std():.4f}")

# 4) Spot-check alignment: compare the first 3 row hashes from each file
def rowhash(v): return hashlib.sha256(v.tobytes()).hexdigest()[:12]
print("\n=== First 3 row hashes (alignment sanity) ===")
for f,x in zip(files, arrays):
    print(os.path.basename(f), [rowhash(x[k]) for k in range(min(3, len(x)))])
