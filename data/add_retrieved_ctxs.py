#!/usr/bin/env python
"""
Add retrieved_ctxs field to training file from merged retriever outputs.
The retrieved_ctxs field will have the same structure as ground_truths and positive_ctxs.
"""

import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm


def read_jsonl(file_path: str) -> List[Dict]:
    """Read a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def read_json(file_path: str) -> List[Dict]:
    """Read a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_jsonl(data: List[Dict], file_path: str):
    """Write list of dictionaries to JSONL file (one compact object per line)."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def add_retrieved_ctxs(
    training_file: str,
    merged_retrievers_file: str,
    output_file: str
) -> Dict:
    """
    Add retrieved_ctxs field to training file from merged retriever outputs.
    """
    
    print(f"\nLoading training file: {training_file}")
    training_data = read_jsonl(training_file)
    print(f"  Loaded {len(training_data)} training examples")
    
    print(f"\nLoading merged retrievers file: {merged_retrievers_file}")
    # Try both JSON and JSONL formats
    try:
        merged_data = read_json(merged_retrievers_file)
        print(f"  Loaded {len(merged_data)} queries (JSON format)")
    except:
        merged_data = read_jsonl(merged_retrievers_file)
        print(f"  Loaded {len(merged_data)} queries (JSONL format)")
    
    # Verify lengths match
    if len(training_data) != len(merged_data):
        print(f"\nWARNING: Length mismatch!")
        print(f"  Training file: {len(training_data)} examples")
        print(f"  Merged file: {len(merged_data)} examples")
        min_len = min(len(training_data), len(merged_data))
        print(f"  Will only process first {min_len} examples")
        training_data = training_data[:min_len]
        merged_data = merged_data[:min_len]
    
    # Statistics
    stats = {
        'n_examples': len(training_data),
        'total_retrieved_docs': 0,
        'examples_with_retrieved': 0,
        'examples_without_retrieved': 0,
        'avg_retrieved_per_example': 0
    }
    
    print(f"\nAdding retrieved_ctxs field to training data...")
    updated_data = []
    
    for idx in tqdm(range(len(training_data))):
        train_example = training_data[idx]
        merged_example = merged_data[idx]
        
        # Verify questions match (optional sanity check)
        train_q = train_example.get('question_text', train_example.get('question', ''))
        merged_q = merged_example.get('question', merged_example.get('input', ''))
        
        if train_q != merged_q:
            print(f"\nWARNING: Question mismatch at index {idx}")
            print(f"  Training: {train_q[:100]}...")
            print(f"  Merged: {merged_q[:100]}...")
        
        # Convert merged ctxs to the same format as ground_truths/positive_ctxs
        # Format: [{"id": "...", "title": "...", "text": "...", "score": ...}, ...]
        retrieved_ctxs = []
        for ctx in merged_example.get('ctxs', []):
            retrieved_doc = {
                'id': ctx.get('id', ''),
                'title': ctx.get('title', ''),
                'text': ctx.get('text', ''),
                'score': ctx.get('score', 0.0)
            }
            retrieved_ctxs.append(retrieved_doc)
        
        # Add retrieved_ctxs to training example
        train_example['retrieved_ctxs'] = retrieved_ctxs
        
        # Update stats
        stats['total_retrieved_docs'] += len(retrieved_ctxs)
        if len(retrieved_ctxs) > 0:
            stats['examples_with_retrieved'] += 1
        else:
            stats['examples_without_retrieved'] += 1
        
        updated_data.append(train_example)
    
    stats['avg_retrieved_per_example'] = (
        stats['total_retrieved_docs'] / stats['n_examples'] 
        if stats['n_examples'] > 0 else 0
    )
    
    print(f"\nSaving updated training file to: {output_file}")
    write_jsonl(updated_data, output_file)
    
    return stats


def print_statistics(stats: Dict):
    """Print statistics."""
    
    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)
    
    print(f"\nTotal examples: {stats['n_examples']:,}")
    print(f"Examples with retrieved docs: {stats['examples_with_retrieved']:,}")
    print(f"Examples without retrieved docs: {stats['examples_without_retrieved']:,}")
    print(f"Total retrieved docs: {stats['total_retrieved_docs']:,}")
    print(f"Avg retrieved docs per example: {stats['avg_retrieved_per_example']:.2f}")


def main():
    # ========================================================================
    # CONFIGURATION - UPDATE THESE PATHS
    # ========================================================================
    
    training_file = '/scratch/dq2024/diverse_retriever/train_data.jsonl'
    merged_retrievers_file = '/scratch/dq2024/diverse_retriever/data/all_model_training_file_all_golds_dedup_answers.json'  # Output from merge_ctxs.py
    output_file = '/scratch/dq2024/diverse_retriever/train_data_with_retrieved.jsonl'
    
    # ========================================================================
    # Process
    # ========================================================================
    
    stats = add_retrieved_ctxs(training_file, merged_retrievers_file, output_file)
    
    # Print statistics
    print_statistics(stats)
    
    # ========================================================================
    # Verification
    # ========================================================================
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    
    print(f"\nReloading updated file...")
    updated = read_jsonl(output_file)
    print(f"✓ Updated file has {len(updated)} examples")
    
    # Check first example
    print(f"\nSample example 0:")
    print(f"  Question: {updated[0].get('question_text', '')[:100]}...")
    print(f"  # ground_truths: {len(updated[0].get('ground_truths', []))}")
    print(f"  # positive_ctxs: {len(updated[0].get('positive_ctxs', []))}")
    print(f"  # retrieved_ctxs: {len(updated[0].get('retrieved_ctxs', []))}")
    
    if updated[0].get('retrieved_ctxs'):
        print(f"\n  First retrieved doc:")
        first_doc = updated[0]['retrieved_ctxs'][0]
        print(f"    id: {first_doc.get('id', '')}")
        print(f"    title: {first_doc.get('title', '')[:50]}...")
        print(f"    text: {first_doc.get('text', '')[:100]}...")
        print(f"    score: {first_doc.get('score', 0)}")
    
    print(f"\n✓ Output saved to: {output_file}")


if __name__ == "__main__":
    main()