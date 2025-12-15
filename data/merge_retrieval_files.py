#!/usr/bin/env python
"""
Merge ctxs (retrieved documents) from multiple retriever output files.
All files have identical questions but different retrieved documents.
Output: Same structure with merged + deduplicated ctxs per query.
"""

import json
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
from tqdm import tqdm


def read_json(file_path: str) -> List[Dict]:
    """Read a JSON file (array of objects)."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(data: List[Dict], file_path: str):
    """Write list of dictionaries to JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def merge_retriever_outputs(
    retriever_files: List[str],
    output_file: str
) -> Dict:
    """
    Merge ctxs from multiple retriever files with identical questions.
    Deduplicates documents per query using document ID.
    """
    
    print(f"\nLoading {len(retriever_files)} retriever files...")
    
    # Load all files
    all_data = []
    for i, path in enumerate(retriever_files):
        print(f"  Loading file {i+1}: {path}")
        data = read_json(path)
        all_data.append(data)
        print(f"    Loaded {len(data)} queries")
    
    # Verify all files have same number of queries
    n_queries = len(all_data[0])
    for i, data in enumerate(all_data[1:], 1):
        if len(data) != n_queries:
            raise ValueError(f"File {i+1} has {len(data)} queries, expected {n_queries}")
    
    print(f"\n✓ All files have {n_queries} queries")
    
    # Statistics
    stats = {
        'n_queries': n_queries,
        'n_retrievers': len(retriever_files),
        'total_docs_before_dedup': 0,
        'total_docs_after_dedup': 0,
        'docs_per_retriever': defaultdict(int),
        'duplicates_removed': 0
    }
    
    # Merge
    print(f"\nMerging ctxs for {n_queries} queries...")
    merged_data = []
    
    for query_idx in tqdm(range(n_queries)):
        # Get base query info from first file
        base_query = all_data[0][query_idx]
        
        # Merge ctxs from all files, deduplicate by doc ID
        seen_doc_ids = set()
        merged_ctxs = []
        
        for file_idx, data in enumerate(all_data):
            query_data = data[query_idx]
            
            # Verify questions match
            if query_data['question'] != base_query['question']:
                raise ValueError(
                    f"Query mismatch at index {query_idx}!\n"
                    f"  File 0: {base_query['question']}\n"
                    f"  File {file_idx}: {query_data['question']}"
                )
            
            # Add ctxs, deduplicating by ID
            for ctx in query_data.get('ctxs', []):
                doc_id = ctx.get('id', '')
                
                stats['total_docs_before_dedup'] += 1
                stats['docs_per_retriever'][f'retriever_{file_idx}'] += 1
                
                if doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    merged_ctxs.append(ctx)
                    stats['total_docs_after_dedup'] += 1
                else:
                    stats['duplicates_removed'] += 1
        
        # Create merged query object
        merged_query = {
            'question': base_query['question'],
            'input': base_query.get('input', base_query['question']),
            'id': base_query.get('id', query_idx),
            'answers': base_query.get('answers', []),
            'ctxs': merged_ctxs,
            'input_negative_ctxs': base_query.get('input_negative_ctxs', [])
        }
        
        merged_data.append(merged_query)
    
    # Save
    print(f"\nSaving merged data to: {output_file}")
    write_json(merged_data, output_file)
    
    return stats


def print_statistics(stats: Dict):
    """Print merge statistics."""
    
    print("\n" + "=" * 70)
    print("MERGE STATISTICS")
    print("=" * 70)
    
    print(f"\nQueries: {stats['n_queries']:,}")
    print(f"Retrievers: {stats['n_retrievers']}")
    
    print(f"\nDocuments per retriever (before dedup):")
    for retriever, count in sorted(stats['docs_per_retriever'].items()):
        print(f"  {retriever}: {count:,}")
    
    print(f"\nTotal docs before dedup: {stats['total_docs_before_dedup']:,}")
    print(f"Total docs after dedup: {stats['total_docs_after_dedup']:,}")
    print(f"Duplicates removed: {stats['duplicates_removed']:,}")
    
    dedup_rate = (stats['duplicates_removed'] / stats['total_docs_before_dedup'] * 100 
                  if stats['total_docs_before_dedup'] > 0 else 0)
    print(f"Deduplication rate: {dedup_rate:.2f}%")
    
    avg_docs_per_query = stats['total_docs_after_dedup'] / stats['n_queries']
    print(f"\nAvg docs per query (after dedup): {avg_docs_per_query:.1f}")


def main():
    # ========================================================================
    # CONFIGURATION - UPDATE THESE PATHS
    # ========================================================================
    
    retriever_files = [
        '/scratch/dq2024/diverse_retriever/data/qwen3_training_file_retrieved_all_golds.json',
        '/scratch/dq2024/diverse_retriever/data/infly_training_file_retrieved_all_golds.json',
        '/scratch/dq2024/diverse_retriever/data/contriever_training_file_retrieved_all_golds.json',
    ]
    
    output_file = '/scratch/dq2024/diverse_retriever/data/all_model_training_file_all_golds.json'
    
    # ========================================================================
    # Merge
    # ========================================================================
    
    stats = merge_retriever_outputs(retriever_files, output_file)
    
    # Print statistics
    print_statistics(stats)
    
    # ========================================================================
    # Verification
    # ========================================================================
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    
    print(f"\nReloading merged file...")
    merged = read_json(output_file)
    print(f"✓ Merged file has {len(merged)} queries")
    
    # Check a few queries
    print(f"\nSample query 0:")
    print(f"  Question: {merged[0]['question'][:100]}...")
    print(f"  Num ctxs: {len(merged[0]['ctxs'])}")
    
    if len(merged) > 1:
        print(f"\nSample query 1:")
        print(f"  Question: {merged[1]['question'][:100]}...")
        print(f"  Num ctxs: {len(merged[1]['ctxs'])}")
    
    print(f"\n✓ Output saved to: {output_file}")


if __name__ == "__main__":
    main()