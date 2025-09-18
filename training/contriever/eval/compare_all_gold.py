#!/usr/bin/env python3
"""
Document Verifier Evaluation Tool

This script evaluates GPT-5 verifier performance against Oracle verifier using 
precision, recall, and accuracy metrics.
"""

import json
import sys
from typing import Dict, List, Set


def load_json_file(filepath: str) -> List[Dict]:
    """Load JSON file and return list of objects."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f if line.strip()]
        return data
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {filepath}: {e}")
        sys.exit(1)


def extract_document_ids(contexts: List[Dict]) -> Set[str]:
    """Extract document IDs from contexts list."""
    return {ctx['id'] for ctx in contexts}


def calculate_metrics(oracle_ids: Set[str], gpt5_ids: Set[str], universe_ids: Set[str]) -> Dict[str, float]:
    """Calculate precision, recall, and accuracy."""
    tp = len(oracle_ids.intersection(gpt5_ids))
    fp = len(gpt5_ids - oracle_ids)
    fn = len(oracle_ids - gpt5_ids)
    tn = len(universe_ids - oracle_ids - gpt5_ids)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    
    return {'precision': precision, 'recall': recall, 'accuracy': accuracy}


def evaluate_verifiers(original_file: str, oracle_file: str, gpt5_file: str) -> None:
    """Main evaluation function."""
    original_data = load_json_file(original_file)
    oracle_data = load_json_file(oracle_file) 
    gpt5_data = load_json_file(gpt5_file)
    
    # Create lookup dictionaries
    original_dict = {item['id']: item for item in original_data}
    oracle_dict = {item['id']: item for item in oracle_data}
    gpt5_dict = {item['id']: item for item in gpt5_data}
    
    # Find common question IDs
    common_ids = set(original_dict.keys()).intersection(
        set(oracle_dict.keys())
    ).intersection(set(gpt5_dict.keys()))
    
    all_metrics = []
    
    for qid in common_ids:
        original_item = original_dict[qid]
        oracle_item = oracle_dict[qid]  
        gpt5_item = gpt5_dict[qid]
        
        # Extract document ID sets
        universe_ids = extract_document_ids(original_item['ctxs'])
        oracle_ids = extract_document_ids(oracle_item['ctxs'])
        gpt5_ids = extract_document_ids(gpt5_item['ctxs'])
        
        # Calculate metrics
        metrics = calculate_metrics(oracle_ids, gpt5_ids, universe_ids)
        all_metrics.append(metrics)
    
    # Calculate macro averages (per query then average)
    macro_precision = sum(m['precision'] for m in all_metrics) / len(all_metrics)
    macro_recall = sum(m['recall'] for m in all_metrics) / len(all_metrics)
    macro_accuracy = sum(m['accuracy'] for m in all_metrics) / len(all_metrics)
    
    # Calculate micro averages (aggregate then calculate)
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    
    for qid in common_ids:
        original_item = original_dict[qid]
        oracle_item = oracle_dict[qid]  
        gpt5_item = gpt5_dict[qid]
        
        universe_ids = extract_document_ids(original_item['ctxs'])
        oracle_ids = extract_document_ids(oracle_item['ctxs'])
        gpt5_ids = extract_document_ids(gpt5_item['ctxs'])
        
        tp = len(oracle_ids.intersection(gpt5_ids))
        fp = len(gpt5_ids - oracle_ids)
        fn = len(oracle_ids - gpt5_ids)
        tn = len(universe_ids - oracle_ids - gpt5_ids)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn
    
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn) if (total_tp + total_fp + total_fn + total_tn) > 0 else 0.0
    
    print("MACRO AVERAGE:")
    print(f"Precision: {macro_precision:.4f}")
    print(f"Recall: {macro_recall:.4f}")
    print(f"Accuracy: {macro_accuracy:.4f}")
    print()
    print("MICRO AVERAGE:")
    print(f"Precision: {micro_precision:.4f}")
    print(f"Recall: {micro_recall:.4f}")
    print(f"Accuracy: {micro_accuracy:.4f}")


def main():
    original_file = '/scratch/dq2024/diverse_retriever/training/contriever/eval/gru/all_golds/qampari_corpus_base_finetuned_only_random_inp0_t100.json'
    oracle_file = '/scratch/dq2024/diverse_retriever/training/contriever/eval/gru/all_golds/base_AR_gold_docs_100.json'
    gpt5_file = '/scratch/dq2024/diverse_retriever/training/contriever/eval/standard/gpt5_verifier/qampari_base_gpt5_nano_all_gold_docs_100.json'
    
    evaluate_verifiers(original_file, oracle_file, gpt5_file)


if __name__ == "__main__":
    main()