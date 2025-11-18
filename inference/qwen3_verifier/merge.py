# merge.py
#!/usr/bin/env python3
"""
Script to merge retrieved documents with ground truth documents.
Takes all ground truth documents and fills remaining slots (up to 100 total)
with highest-scoring non-duplicate documents from retrieved set.
"""

import json
import sys
from typing import Dict, List, Set, Any


def extract_doc_ids(ctxs: List[Dict]) -> Set[str]:
    """Extract document IDs from contexts list."""
    return {ctx.get('id', '') for ctx in ctxs if ctx.get('id')}


def merge_contexts(retrieved_ctxs: List[Dict], ground_truth_ctxs: List[Dict], 
                  max_total: int = 100) -> List[Dict]:
    """
    Merge contexts by taking all ground truth docs and filling remaining slots
    with highest-scoring non-duplicate retrieved docs.
    
    Args:
        retrieved_ctxs: List of retrieved documents with scores
        ground_truth_ctxs: List of ground truth documents
        max_total: Maximum total documents to return (default 100)
    
    Returns:
        Merged list of contexts
    """
    # Start with all ground truth documents
    merged = ground_truth_ctxs.copy()
    
    # Get IDs of documents already in ground truth
    used_ids = extract_doc_ids(ground_truth_ctxs)
    
    # If we already have max_total or more docs, return truncated list
    if len(merged) >= max_total:
        return merged[:max_total]
    
    # Sort retrieved contexts by score (descending)
    # Assumes 'score' field exists and higher is better
    sorted_retrieved = sorted(
        retrieved_ctxs,
        key=lambda x: float(x.get('score', 0)),
        reverse=True
    )
    
    # Add non-duplicate retrieved docs until we reach max_total
    for ctx in sorted_retrieved:
        if len(merged) >= max_total:
            break
            
        doc_id = ctx.get('id', '')
        if doc_id and doc_id not in used_ids:
            merged.append(ctx)
            used_ids.add(doc_id)
    
    return merged


def process_files(retrieved_file: str, ground_truth_file: str, output_file: str):
    """
    Process the two input files and create merged output file.
    
    Args:
        retrieved_file: Path to file with retrieved documents
        ground_truth_file: Path to file with ground truth documents
        output_file: Path to output file
    """
    print(f"Reading retrieved documents from: {retrieved_file}")
    print(f"Reading ground truth documents from: {ground_truth_file}")
    print(f"Writing merged output to: {output_file}")
    
    with open(retrieved_file, 'r', encoding='utf-8') as f_ret, \
         open(ground_truth_file, 'r', encoding='utf-8') as f_gt, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        query_count = 0
        
        while True:
            # Read one line from each file
            ret_line = f_ret.readline()
            gt_line = f_gt.readline()
            
            # Check if we've reached end of either file
            if not ret_line or not gt_line:
                break
            
            # Parse JSON objects
            try:
                ret_obj = json.loads(ret_line.strip())
                gt_obj = json.loads(gt_line.strip())
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON at query {query_count + 1}: {e}")
                continue
            
            # Verify questions match (optional safety check)
            if ret_obj.get('question') != gt_obj.get('question'):
                print(f"Warning: Questions don't match at query {query_count + 1}")
                print(f"  Retrieved: {ret_obj.get('question', 'N/A')}")
                print(f"  Ground truth: {gt_obj.get('question', 'N/A')}")
            
            # Merge contexts
            merged_ctxs = merge_contexts(
                ret_obj.get('ctxs', []),
                gt_obj.get('ctxs', []),
                max_total=100
            )
            
            # Create output object (copy retrieved object and update ctxs)
            output_obj = ret_obj.copy()
            output_obj['ctxs'] = merged_ctxs
            
            # Write to output file
            f_out.write(json.dumps(output_obj, ensure_ascii=False) + '\n')
            
            query_count += 1
            
            # Progress indicator
            if query_count % 100 == 0:
                print(f"Processed {query_count} queries...")
        
        print(f"Completed processing {query_count} queries")


def main():
    """Main function to handle command line arguments and run processing."""
    # File names as string variables
    
    ground_truth_file = "/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/contriever/qampari_base_qwen3-30B_all_gold_docs_500_vllm.json"
    retrieved_file = "/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/contriever/qampari_std_t500_promote-qwen3_3_retrieved.json"
    output_file = "/scratch/dq2024/diverse_retriever/eval/contriever/qwen3_verifier/MERGED_qampari_std_t500_promote-qwen3_3_retrieved.json"
    process_files(retrieved_file, ground_truth_file, output_file)
    print(f"Successfully created merged file: {output_file}")

    ground_truth_file = "/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/qwen3/qampari_base_qwen3-30B_all_gold_docs_200.json"
    retrieved_file = "/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/qwen3/qampari_std_t200_promote-qwen3_0-3_retrieved.json"
    output_file = "/scratch/dq2024/diverse_retriever/eval/qwen3/qwen3_verifier/MERGED_qampari_std_t200_promote-qwen3_0-3_retrieved.json"
    process_files(retrieved_file, ground_truth_file, output_file)
    print(f"Successfully created merged file: {output_file}")

    



if __name__ == "__main__":
    main()