#!/usr/bin/env python3
"""
Gold Document Comparison Tool

This script compares two JSON files to evaluate how well the GPT-5 verifier 
performed in retrieving correct gold documents compared to ground truth.
"""

import json
import sys
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple


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


def calculate_metrics(ground_truth_ids: Set[str], predicted_ids: Set[str]) -> Dict[str, float]:
    """Calculate precision, recall, and F1 score."""
    if not predicted_ids:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    if not ground_truth_ids:
        return {'precision': 0.0 if predicted_ids else 1.0, 'recall': 1.0, 'f1': 0.0}
    
    intersection = ground_truth_ids.intersection(predicted_ids)
    
    precision = len(intersection) / len(predicted_ids) if predicted_ids else 0.0
    recall = len(intersection) / len(ground_truth_ids) if ground_truth_ids else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def analyze_overlap_patterns(ground_truth_ids: Set[str], predicted_ids: Set[str]) -> Dict[str, int]:
    """Analyze overlap patterns between ground truth and predicted documents."""
    intersection = ground_truth_ids.intersection(predicted_ids)
    only_ground_truth = ground_truth_ids - predicted_ids
    only_predicted = predicted_ids - ground_truth_ids
    
    return {
        'common_docs': len(intersection),
        'missed_docs': len(only_ground_truth),
        'extra_docs': len(only_predicted),
        'total_ground_truth': len(ground_truth_ids),
        'total_predicted': len(predicted_ids)
    }


def compare_documents(ground_truth_file: str, predicted_file: str) -> None:
    """Main comparison function."""
    print("Loading files...")
    ground_truth_data = load_json_file(ground_truth_file)
    predicted_data = load_json_file(predicted_file)
    
    print(f"Ground truth entries: {len(ground_truth_data)}")
    print(f"Predicted entries: {len(predicted_data)}")
    
    # Create dictionaries for quick lookup by question ID
    gt_dict = {item['id']: item for item in ground_truth_data}
    pred_dict = {item['id']: item for item in predicted_data}
    
    # Find common question IDs
    common_ids = set(gt_dict.keys()).intersection(set(pred_dict.keys()))
    print(f"Common question IDs: {len(common_ids)}")
    
    if not common_ids:
        print("No common question IDs found between files!")
        return
    
    # Initialize metrics storage
    all_metrics = []
    overlap_stats = []
    question_analysis = []
    
    perfect_matches = 0
    no_overlap = 0
    
    print("\nAnalyzing each question...")
    
    for qid in sorted(common_ids):
        gt_item = gt_dict[qid]
        pred_item = pred_dict[qid]
        
        # Extract document IDs
        gt_doc_ids = extract_document_ids(gt_item['ctxs'])
        pred_doc_ids = extract_document_ids(pred_item['ctxs'])
        
        # Calculate metrics
        metrics = calculate_metrics(gt_doc_ids, pred_doc_ids)
        all_metrics.append(metrics)
        
        # Analyze overlap
        overlap = analyze_overlap_patterns(gt_doc_ids, pred_doc_ids)
        overlap_stats.append(overlap)
        
        # Track perfect matches and no overlap
        if metrics['f1'] == 1.0:
            perfect_matches += 1
        if metrics['f1'] == 0.0:
            no_overlap += 1
        
        # Store detailed analysis
        question_analysis.append({
            'question_id': qid,
            'question': gt_item.get('question', 'N/A'),
            'ground_truth_docs': len(gt_doc_ids),
            'predicted_docs': len(pred_doc_ids),
            'common_docs': overlap['common_docs'],
            'missed_docs': overlap['missed_docs'],
            'extra_docs': overlap['extra_docs'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1']
        })
    
    # Calculate overall statistics
    avg_metrics = {
        'precision': sum(m['precision'] for m in all_metrics) / len(all_metrics),
        'recall': sum(m['recall'] for m in all_metrics) / len(all_metrics),
        'f1': sum(m['f1'] for m in all_metrics) / len(all_metrics)
    }
    
    avg_overlap = {
        'common_docs': sum(o['common_docs'] for o in overlap_stats) / len(overlap_stats),
        'missed_docs': sum(o['missed_docs'] for o in overlap_stats) / len(overlap_stats),
        'extra_docs': sum(o['extra_docs'] for o in overlap_stats) / len(overlap_stats),
        'total_ground_truth': sum(o['total_ground_truth'] for o in overlap_stats) / len(overlap_stats),
        'total_predicted': sum(o['total_predicted'] for o in overlap_stats) / len(overlap_stats)
    }
    
    # Print results
    print("="*80)
    print("OVERALL PERFORMANCE METRICS")
    print("="*80)
    print(f"Total questions analyzed: {len(common_ids)}")
    print(f"Perfect matches (F1=1.0): {perfect_matches} ({perfect_matches/len(common_ids)*100:.1f}%)")
    print(f"No overlap (F1=0.0): {no_overlap} ({no_overlap/len(common_ids)*100:.1f}%)")
    print()
    print("Average Metrics:")
    print(f"  Precision: {avg_metrics['precision']:.4f}")
    print(f"  Recall:    {avg_metrics['recall']:.4f}")
    print(f"  F1 Score:  {avg_metrics['f1']:.4f}")
    print()
    print("Average Document Counts:")
    print(f"  Ground Truth Docs per Question: {avg_overlap['total_ground_truth']:.1f}")
    print(f"  Predicted Docs per Question:    {avg_overlap['total_predicted']:.1f}")
    print(f"  Common Docs per Question:       {avg_overlap['common_docs']:.1f}")
    print(f"  Missed Docs per Question:       {avg_overlap['missed_docs']:.1f}")
    print(f"  Extra Docs per Question:        {avg_overlap['extra_docs']:.1f}")
    
    # Distribution analysis
    print("\n" + "="*80)
    print("PERFORMANCE DISTRIBUTION")
    print("="*80)
    
    f1_ranges = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0), (1.0, 1.0)]
    f1_counts = Counter()
    
    for metrics in all_metrics:
        f1_score = metrics['f1']
        for range_start, range_end in f1_ranges:
            if range_start <= f1_score <= range_end:
                if range_start == 1.0:  # Perfect score
                    f1_counts['1.0 (Perfect)'] += 1
                else:
                    f1_counts[f'{range_start:.1f}-{range_end:.1f}'] += 1
                break
    
    print("F1 Score Distribution:")
    for range_name in ['0.0-0.1', '0.1-0.3', '0.3-0.5', '0.5-0.7', '0.7-0.9', '0.9-1.0', '1.0 (Perfect)']:
        count = f1_counts[range_name]
        percentage = count / len(all_metrics) * 100
        print(f"  {range_name}: {count:3d} questions ({percentage:5.1f}%)")
    
    # Top worst performing questions
    print("\n" + "="*80)
    print("WORST PERFORMING QUESTIONS (Bottom 10 by F1 Score)")
    print("="*80)
    
    worst_questions = sorted(question_analysis, key=lambda x: x['f1'])[:10]
    for i, qa in enumerate(worst_questions, 1):
        print(f"{i:2d}. ID: {qa['question_id']} | F1: {qa['f1']:.3f} | "
              f"GT: {qa['ground_truth_docs']} docs | Pred: {qa['predicted_docs']} docs | "
              f"Common: {qa['common_docs']}")
        print(f"    Question: {qa['question'][:100]}{'...' if len(qa['question']) > 100 else ''}")
        print()
    
    # Top best performing questions  
    print("="*80)
    print("BEST PERFORMING QUESTIONS (Top 10 by F1 Score)")
    print("="*80)
    
    best_questions = sorted(question_analysis, key=lambda x: x['f1'], reverse=True)[:10]
    for i, qa in enumerate(best_questions, 1):
        print(f"{i:2d}. ID: {qa['question_id']} | F1: {qa['f1']:.3f} | "
              f"GT: {qa['ground_truth_docs']} docs | Pred: {qa['predicted_docs']} docs | "
              f"Common: {qa['common_docs']}")
        print(f"    Question: {qa['question'][:100]}{'...' if len(qa['question']) > 100 else ''}")
        print()


def main():
    # Hardcoded file paths
    ground_truth_file = '/scratch/dq2024/diverse_retriever/training/contriever/eval/gru/all_golds/base_AR_gold_docs_100.json'
    predicted_file = '/scratch/dq2024/diverse_retriever/training/contriever/eval/standard/gpt5_verifier/qampari_base_gpt5_nano_all_gold_docs_100.json'
    
    compare_documents(ground_truth_file, predicted_file)


if __name__ == "__main__":
    main()