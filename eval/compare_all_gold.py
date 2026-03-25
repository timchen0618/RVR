# #!/usr/bin/env python3
# """
# Document Verifier Evaluation Tool

# This script evaluates GPT-5 verifier performance against Oracle verifier using 
# precision, recall, and accuracy metrics.
# """

# import json
# import sys
# from typing import Dict, List, Set


# def load_json_file(filepath: str) -> List[Dict]:
#     """Load JSON file and return list of objects."""
#     try:
#         with open(filepath, 'r', encoding='utf-8') as f:
#             data = [json.loads(line.strip()) for line in f if line.strip()]
#         return data
#     except FileNotFoundError:
#         print(f"Error: File {filepath} not found.")
#         sys.exit(1)
#     except json.JSONDecodeError as e:
#         print(f"Error: Invalid JSON in {filepath}: {e}")
#         sys.exit(1)


# def extract_document_ids(contexts: List[Dict]) -> Set[str]:
#     """Extract document IDs from contexts list."""
#     return {ctx['id'] for ctx in contexts}


# def calculate_metrics(oracle_ids: Set[str], gpt5_ids: Set[str], universe_ids: Set[str]) -> Dict[str, float]:
#     """Calculate precision, recall, and accuracy."""
#     tp = len(oracle_ids.intersection(gpt5_ids))
#     fp = len(gpt5_ids - oracle_ids)
#     fn = len(oracle_ids - gpt5_ids)
#     tn = len(universe_ids - oracle_ids - gpt5_ids)
    
#     precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#     recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#     accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    
#     return {'precision': precision, 'recall': recall, 'accuracy': accuracy}


# def evaluate_verifiers(original_file: str, oracle_file: str, gpt5_file: str) -> None:
#     """Main evaluation function."""
#     original_data = load_json_file(original_file)
#     oracle_data = load_json_file(oracle_file) 
#     gpt5_data = load_json_file(gpt5_file)
    
#     # Create lookup dictionaries
#     original_dict = {item['id']: item for item in original_data}
#     oracle_dict = {item['id']: item for item in oracle_data}
#     gpt5_dict = {item['id']: item for item in gpt5_data}
    
#     # Find common question IDs
#     common_ids = set(original_dict.keys()).intersection(
#         set(oracle_dict.keys())
#     ).intersection(set(gpt5_dict.keys()))
    
#     all_metrics = []
#     total_positive = 0
#     total_negative = 0
    
#     for qid in common_ids:
#         original_item = original_dict[qid]
#         oracle_item = oracle_dict[qid]  
#         gpt5_item = gpt5_dict[qid]
        
#         # Extract document ID sets
#         universe_ids = extract_document_ids(original_item['ctxs'])
#         oracle_ids = extract_document_ids(oracle_item['ctxs'])
#         gpt5_ids = extract_document_ids(gpt5_item['ctxs'])
        
#         # Count positive/negative in universe
#         positive_docs = len(oracle_ids)  # Oracle determines ground truth
#         negative_docs = len(universe_ids) - positive_docs
        
#         total_positive += positive_docs
#         total_negative += negative_docs
        
#         # Calculate metrics
#         metrics = calculate_metrics(oracle_ids, gpt5_ids, universe_ids)
#         all_metrics.append(metrics)
    
#     # Calculate averages
#     avg_precision = sum(m['precision'] for m in all_metrics) / len(all_metrics)
#     avg_recall = sum(m['recall'] for m in all_metrics) / len(all_metrics)
#     avg_accuracy = sum(m['accuracy'] for m in all_metrics) / len(all_metrics)
    
#     # Calculate class distribution
#     total_docs = total_positive + total_negative
#     positive_percentage = (total_positive / total_docs) * 100
#     negative_percentage = (total_negative / total_docs) * 100
    
#     print(f"Precision: {avg_precision:.4f}")
#     print(f"Recall: {avg_recall:.4f}")
#     print(f"Accuracy: {avg_accuracy:.4f}")
#     print(f"Positive documents: {positive_percentage:.2f}%")
#     print(f"Negative documents: {negative_percentage:.2f}%")


# def main():
#     original_file = '/scratch/dq2024/diverse_retriever/eval/contriever/base/qampari_corpus_finetuned_base_t100.json'
#     oracle_file = '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/v100/infly/max6/base/gold_documents/round1_all_gold_docs.json'
#     gpt5_file = '/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/infly/qampari_infly_base_qwen3-4B_all_gold_docs_100.json'
    
#     evaluate_verifiers(original_file, oracle_file, gpt5_file)


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
"""
Document Verifier Evaluation Tool

This script evaluates GPT-5 verifier performance against Oracle verifier using 
precision, recall, and accuracy metrics.
"""

import json
import sys
import random
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


def calculate_metrics(oracle_ids: Set[str], predicted_ids: Set[str], universe_ids: Set[str]) -> Dict[str, float]:
    """Calculate precision, recall, and accuracy."""
    tp = len(oracle_ids.intersection(predicted_ids))
    fp = len(predicted_ids - oracle_ids)
    fn = len(oracle_ids - predicted_ids)
    tn = len(universe_ids - oracle_ids - predicted_ids)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    
    return {'precision': precision, 'recall': recall, 'accuracy': accuracy}


def generate_random_predictions(universe_ids: Set[str], positive_rate: float, seed: int = 42) -> Set[str]:
    """
    Generate random predictions based on positive rate.
    
    Args:
        universe_ids: Set of all document IDs
        positive_rate: Probability of labeling a document as positive
        seed: Random seed for reproducibility
    
    Returns:
        Set of randomly selected document IDs predicted as positive
    """
    random.seed(seed)
    return {doc_id for doc_id in universe_ids if random.random() < positive_rate}


def evaluate_verifiers(original_file: str, oracle_file: str, gpt5_file: str, 
                       include_random_baseline: bool = True, num_random_trials: int = 100) -> None:
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
    
    gpt5_metrics = []
    random_metrics_all_trials = []
    total_positive = 0
    total_negative = 0
    
    # First pass: calculate positive rate from oracle
    for qid in common_ids:
        oracle_item = oracle_dict[qid]
        original_item = original_dict[qid]
        
        universe_ids = extract_document_ids(original_item['ctxs'])
        oracle_ids = extract_document_ids(oracle_item['ctxs'])
        
        positive_docs = len(oracle_ids)
        negative_docs = len(universe_ids) - positive_docs
        
        total_positive += positive_docs
        total_negative += negative_docs
    
    total_docs = total_positive + total_negative
    positive_rate = total_positive / total_docs
    
    print(f"Overall positive rate: {positive_rate:.4f} ({positive_rate*100:.2f}%)")
    print(f"Total positive documents: {total_positive}")
    print(f"Total negative documents: {total_negative}")
    print()
    
    # Second pass: evaluate GPT-5 and random baseline
    for qid in common_ids:
        original_item = original_dict[qid]
        oracle_item = oracle_dict[qid]  
        gpt5_item = gpt5_dict[qid]
        
        # Extract document ID sets
        universe_ids = extract_document_ids(original_item['ctxs'])
        oracle_ids = extract_document_ids(oracle_item['ctxs'])
        gpt5_ids = extract_document_ids(gpt5_item['ctxs'])
        
        # Calculate GPT-5 metrics
        gpt5_metric = calculate_metrics(oracle_ids, gpt5_ids, universe_ids)
        gpt5_metrics.append(gpt5_metric)
        
        # Calculate random baseline metrics (multiple trials for stability)
        if include_random_baseline:
            for trial in range(num_random_trials):
                random_ids = generate_random_predictions(universe_ids, positive_rate, seed=trial)
                random_metric = calculate_metrics(oracle_ids, random_ids, universe_ids)
                random_metrics_all_trials.append(random_metric)
    
    # Calculate GPT-5 averages
    avg_gpt5_precision = sum(m['precision'] for m in gpt5_metrics) / len(gpt5_metrics)
    avg_gpt5_recall = sum(m['recall'] for m in gpt5_metrics) / len(gpt5_metrics)
    avg_gpt5_accuracy = sum(m['accuracy'] for m in gpt5_metrics) / len(gpt5_metrics)
    
    print("=" * 60)
    print("GPT-5 VERIFIER PERFORMANCE")
    print("=" * 60)
    print(f"Precision: {avg_gpt5_precision:.4f}")
    print(f"Recall:    {avg_gpt5_recall:.4f}")
    print(f"Accuracy:  {avg_gpt5_accuracy:.4f}")
    print()
    
    if include_random_baseline:
        # Calculate random baseline averages
        avg_random_precision = sum(m['precision'] for m in random_metrics_all_trials) / len(random_metrics_all_trials)
        avg_random_recall = sum(m['recall'] for m in random_metrics_all_trials) / len(random_metrics_all_trials)
        avg_random_accuracy = sum(m['accuracy'] for m in random_metrics_all_trials) / len(random_metrics_all_trials)
        
        print("=" * 60)
        print(f"RANDOM BASELINE (positive_rate={positive_rate:.4f}, {num_random_trials} trials)")
        print("=" * 60)
        print(f"Precision: {avg_random_precision:.4f}")
        print(f"Recall:    {avg_random_recall:.4f}")
        print(f"Accuracy:  {avg_random_accuracy:.4f}")
        print()
        
        # Calculate improvement over random
        print("=" * 60)
        print("IMPROVEMENT OVER RANDOM BASELINE")
        print("=" * 60)
        print(f"Precision: +{(avg_gpt5_precision - avg_random_precision):.4f} ({((avg_gpt5_precision / avg_random_precision - 1) * 100):.1f}% improvement)")
        print(f"Recall:    +{(avg_gpt5_recall - avg_random_recall):.4f} ({((avg_gpt5_recall / avg_random_recall - 1) * 100):.1f}% improvement)")
        print(f"Accuracy:  +{(avg_gpt5_accuracy - avg_random_accuracy):.4f} ({((avg_gpt5_accuracy / avg_random_accuracy - 1) * 100):.1f}% improvement)")


def main():
    # /scratch/dq2024/diverse_retriever/eval/fanoutQA/multi_round_pipeline/qwen3_verifier/contriever/retrieved_files/round1_retrieved.json
    original_file = '/scratch/dq2024/diverse_retriever/eval/contriever/base/qampari_corpus_finetuned_base_t100.json'
    oracle_file = '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/v100/infly/max6/base/gold_documents/round1_all_gold_docs.json'
    gpt5_file = '/scratch/dq2024/diverse_retriever/inference/gpt_verifier/qampari_infly_base_gpt5_nano_all_gold_docs_100.json'
    
    evaluate_verifiers(original_file, oracle_file, gpt5_file, 
                       include_random_baseline=True, num_random_trials=100)
    
    original_file = '/scratch/dq2024/diverse_retriever/eval/contriever/base/qampari_corpus_finetuned_base_t100.json'
    oracle_file = '/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/v100/qwen3/max6/base/gold_documents/round1_all_gold_docs.json'
    gpt5_file = '/scratch/dq2024/diverse_retriever/inference/gpt_verifier/qampari_qwen3_base_gpt5_nano_all_gold_docs_100.json'
    
    evaluate_verifiers(original_file, oracle_file, gpt5_file, 
                       include_random_baseline=True, num_random_trials=100)


if __name__ == "__main__":
    main()