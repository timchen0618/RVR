# compare_query_vectors.py
import numpy as np
import sys
import os

def compute_cosine_similarity(embeddings1, embeddings2):
    """Compute cosine similarity between two sets of embeddings"""
    embeddings1_norm = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2_norm = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
    cosine_sims = np.sum(embeddings1_norm * embeddings2_norm, axis=1)
    return cosine_sims

if __name__ == "__main__":
    
    file1 = '/scratch/dq2024/diverse_retriever/training/contriever/eval/base/query_embeddings__scratch_dq2024_models_diverse_retriever_checkpoints_qampari_base_finetuned_only_random_checkpoint_best_model.npy'
    file2 = '/scratch/dq2024/diverse_retriever/training/contriever/eval/gru/two_stage/query_embeddings_qampari_corpus_only_random_gold_AR_r1contr_r2gru_0123_100.json.npy'
    
    embeddings1 = np.load(file1)
    embeddings2 = np.load(file2)
    
    assert embeddings1.shape == embeddings2.shape, f"Shape mismatch: {embeddings1.shape} vs {embeddings2.shape}"
    
    similarities = compute_cosine_similarity(embeddings1, embeddings2)
    
    print(f"Loaded embeddings from:")
    print(f"  File 1: {file1}")
    print(f"  File 2: {file2}")
    print(f"Shape: {embeddings1.shape}")
    print()
    
    print(f"Cosine similarities (cos(q1,q2)):")
    print(f"  Mean: {np.mean(similarities):.4f}")
    print(f"  Std:  {np.std(similarities):.4f}")
    print(f"  Min:  {np.min(similarities):.4f}")
    print(f"  Max:  {np.max(similarities):.4f}")
    
    