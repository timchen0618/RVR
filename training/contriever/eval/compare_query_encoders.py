#!/usr/bin/env python3
import numpy as np
import argparse

def cos_rows(A, B):
    """Row-wise cosine similarity between two (N,d) arrays."""
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    return np.einsum("ij,ij->i", A, B)

def main(base_path, gru_path, std_path):
    base = np.load(base_path).astype(np.float32)
    gru  = np.load(gru_path).astype(np.float32)
    std  = np.load(std_path).astype(np.float32)

    c_base_gru = cos_rows(base, gru)
    c_base_std = cos_rows(base, std)

    print("=== Cosine stats vs BASE ===")
    print(f"GRU: mean={c_base_gru.mean():.8f}, std={c_base_gru.std():.8f}")
    print(f"STD: mean={c_base_std.mean():.8f}, std={c_base_std.std():.8f}")

    delta = c_base_gru - c_base_std
    print("\n=== GRU – STD delta ===")
    print(f"Δ mean={delta.mean():.8f}, Δ std={delta.std():.8f}, Δ max={np.abs(delta).max():.8f}")

    corr = np.corrcoef(c_base_gru, c_base_std)[0,1]
    print(f"\nCorrelation between (base·gru) and (base·std): {corr:.8f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base", help="Path to base .npy embeddings")
    parser.add_argument("gru", help="Path to GRU .npy embeddings")
    parser.add_argument("std", help="Path to STD .npy embeddings")
    args = parser.parse_args()

    main(args.base, args.gru, args.std)
