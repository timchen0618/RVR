# compare_query_vectors.py
import numpy as np
import sys
import os

def compute_cosine_similarity(embeddings1, embeddings2):
    """Compute cosine similarity between two sets of embeddings"""
    # upcast to float32 to avoid fp16 quantization error
    embeddings1 = embeddings1.astype(np.float32, copy=False)
    embeddings2 = embeddings2.astype(np.float32, copy=False)

    embeddings1_norm = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2_norm = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
    cosine_sims = np.sum(embeddings1_norm * embeddings2_norm, axis=1)
    return cosine_sims

if __name__ == "__main__":
    
    file1 = '/scratch/dq2024/diverse_retriever/training/contriever/eval/base/query_embeddings__scratch_dq2024_models_diverse_retriever_checkpoints_qampari_base_finetuned_only_random_checkpoint_best_model.npy'
    file1 = '/scratch/dq2024/diverse_retriever/eval/qwen3/base/npy/query_embeddings_temp.json.npy'
    files = [
        # '/scratch/dq2024/diverse_retriever/training/contriever/eval/gru/two_stage/query_embeddings_qampari_corpus_only_random_gold_AR_r1contr_r2gru_100.json.npy',
        # '/scratch/dq2024/diverse_retriever/training/contriever/eval/gru/two_stage/query_embeddings_qampari_corpus_only_random_gold_AR_r1contr_r2gru_0123_100.json.npy',
        # '/scratch/dq2024/diverse_retriever/training/contriever/eval/standard/query_embeddings_qampari_corpus_only_random_gold_AR_r1contr_r2std_100.json.npy',
        # '/scratch/dq2024/diverse_retriever/training/contriever/eval/standard/query_embeddings_qampari_corpus_only_random_gold_AR_r1contr_r2std_0123_100.json.npy',

        # '/scratch/dq2024/diverse_retriever/training/contriever/eval/gru/two_stage/query_embeddings_qampari_corpus_only_random_gold_ER_r1contr_r2gru_100.json.npy',
        # '/scratch/dq2024/diverse_retriever/training/contriever/eval/gru/two_stage/query_embeddings_qampari_corpus_only_random_gold_ER_r1contr_r2gru_0123_100.json.npy',
        # '/scratch/dq2024/diverse_retriever/training/contriever/eval/standard/query_embeddings_qampari_corpus_only_random_gold_ER_r1contr_r2std_100.json.npy',
        # '/scratch/dq2024/diverse_retriever/training/contriever/eval/standard/query_embeddings_qampari_corpus_only_random_gold_ER_r1contr_r2std_0123_100.json.npy',

        # '/scratch/dq2024/diverse_retriever/training/contriever/eval/gru/two_stage/query_embeddings_qampari_corpus_only_random_gold_AR_r1contr_r2gru_200.json.npy',
        # '/scratch/dq2024/diverse_retriever/training/contriever/eval/gru/two_stage/query_embeddings_qampari_corpus_only_random_gold_AR_r1contr_r2gru_0123_200.json.npy',
        # '/scratch/dq2024/diverse_retriever/training/contriever/eval/standard/query_embeddings_qampari_corpus_only_random_gold_AR_r1contr_r2std_200.json.npy',
        # '/scratch/dq2024/diverse_retriever/training/contriever/eval/standard/query_embeddings_qampari_corpus_only_random_gold_AR_r1contr_r2std_0123_200.json.npy',

        # '/scratch/dq2024/diverse_retriever/training/contriever/eval/gru/two_stage/query_embeddings_qampari_corpus_only_random_gold_ER_r1contr_r2gru_200.json.npy',
        # '/scratch/dq2024/diverse_retriever/training/contriever/eval/gru/two_stage/query_embeddings_qampari_corpus_only_random_gold_ER_r1contr_r2gru_0123_200.json.npy',
        # '/scratch/dq2024/diverse_retriever/training/contriever/eval/standard/query_embeddings_qampari_corpus_only_random_gold_ER_r1contr_r2std_200.json.npy',
        # '/scratch/dq2024/diverse_retriever/training/contriever/eval/standard/query_embeddings_qampari_corpus_only_random_gold_ER_r1contr_r2std_0123_200.json.npy',

        '/scratch/dq2024/diverse_retriever/inference/qwen3/standard/max12_model/npy/query_embeddings_qampari_corpus_base_r1qw_r2std0-12_AR_0-1_t200.json.npy',
        '/scratch/dq2024/diverse_retriever/inference/qwen3/standard/max12_model/npy/query_embeddings_qampari_corpus_base_r1qw_r2std0-12_AR_0-2_t200.json.npy',
        '/scratch/dq2024/diverse_retriever/inference/qwen3/standard/max12_model/npy/query_embeddings_qampari_corpus_base_r1qw_r2std0-12_AR_0-3_t200.json.npy'
        
    ]
    

    for i in range(len(files)):
        file2 = files[i]
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
        print(f"  Mean: {np.mean(similarities):.8f}")
        print(f"  Std:  {np.std(similarities):.8f}")
        print(f"  Min:  {np.min(similarities):.8f}")
        print(f"  Max:  {np.max(similarities):.8f}")
    
    # gru_standard_pairs = [
    #     ('/scratch/dq2024/diverse_retriever/training/contriever/eval/gru/two_stage/query_embeddings_qampari_corpus_only_random_gold_AR_r1contr_r2gru_0123_100.json.npy',
    #      '/scratch/dq2024/diverse_retriever/training/contriever/eval/standard/query_embeddings_qampari_corpus_only_random_gold_AR_r1contr_r2std_0123_100.json.npy'),
    #     ('/scratch/dq2024/diverse_retriever/training/contriever/eval/gru/two_stage/query_embeddings_qampari_corpus_only_random_gold_AR_r1contr_r2gru_0123_200.json.npy',
    #      '/scratch/dq2024/diverse_retriever/training/contriever/eval/standard/query_embeddings_qampari_corpus_only_random_gold_AR_r1contr_r2std_0123_200.json.npy')
    # ]
    
    # for gru_file, std_file in gru_standard_pairs:
    #     gru_embeddings = np.load(gru_file)
    #     std_embeddings = np.load(std_file)
        
    #     assert gru_embeddings.shape == std_embeddings.shape, f"Shape mismatch: {gru_embeddings.shape} vs {std_embeddings.shape}"
        
    #     direct_similarities = compute_cosine_similarity(gru_embeddings, std_embeddings)
        
    #     print(f"Direct comparison:")
    #     print(f"  GRU file: {os.path.basename(gru_file)}")
    #     print(f"  STD file: {os.path.basename(std_file)}")
    #     print(f"Shape: {gru_embeddings.shape}")
    #     print()
        
    #     print(f"Direct GRU vs Standard cosine similarities:")
    #     print(f"  Mean: {np.mean(direct_similarities):.4f}")
    #     print(f"  Std:  {np.std(direct_similarities):.4f}")
    #     print(f"  Min:  {np.min(direct_similarities):.4f}")
    #     print(f"  Max:  {np.max(direct_similarities):.4f}")
    #     print()
    
    