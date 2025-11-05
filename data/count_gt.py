#!/usr/bin/env python
"""
Count total gold documents in ground truth dataset.
"""

import json
import argparse
from typing import Tuple, List


def read_jsonl(file_path):
    import json
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]


def find_zero_answer_queries(ground_truth_path: str):
    """
    Find and print all queries that have zero answers.
    
    Args:
        ground_truth_path: Path to ground truth JSONL file
    """
    dataset = read_jsonl(ground_truth_path)
    
    zero_answer_queries = []
    
    for idx, item in enumerate(dataset):
        # Check for answer fields
        num_answers = 0
        answer_field = None
        
        if 'answer_list' in item:
            num_answers = len(item['answer_list'])
            answer_field = 'answer_list'
        elif 'answers' in item:
            num_answers = len(item['answers'])
            answer_field = 'answers'
        
        # If zero answers, save it
        if num_answers == 0:
            query_info = {
                'index': idx,
                'question': item.get('question_text', item.get('org_q', item.get('question', 'N/A'))),
                'qid': item.get('qid', f'index_{idx}'),
                'answer_field': answer_field,
                'full_item': item
            }
            zero_answer_queries.append(query_info)
    
    # Print results
    print("="*80)
    print("QUERIES WITH ZERO ANSWERS")
    print("="*80)
    print(f"\nGround truth file: {ground_truth_path}")
    print(f"Total questions: {len(dataset)}")
    print(f"Questions with zero answers: {len(zero_answer_queries)}")
    print(f"Percentage: {len(zero_answer_queries)/len(dataset)*100:.2f}%")
    print("\n" + "="*80)
    
    if zero_answer_queries:
        print("\nDETAILS OF ZERO-ANSWER QUERIES:")
        print("="*80)
        
        for i, query in enumerate(zero_answer_queries, 1):
            print(f"\n[Query {i}]")
            print(f"Index: {query['index']}")
            print(f"QID: {query['qid']}")
            print(f"Question: {query['question']}")
            print(f"Answer field: {query['answer_field']}")
            print(f"\nFull JSON:")
            print(json.dumps(query['full_item'], indent=2))
            print("-"*80)
    else:
        print("\nNo queries with zero answers found.")
    
    return zero_answer_queries

def count_total_gold_documents(ground_truth_path: str, has_gold_id: bool = False) -> Tuple[int, float]:
    """
    Count total number of gold documents across all questions.
    
    Args:
        ground_truth_path: Path to ground truth JSONL file
        has_gold_id: Whether gold document IDs are provided (True) or 
                     we count based on answers (False)
    
    Returns:
        Tuple of (total_gold_docs, avg_gold_per_question)
    """
    dataset = read_jsonl(ground_truth_path)
    
    total_gold_docs = 0
    gold_docs_per_question = []
    questions_with_no_golds = 0
    
    for idx, item in enumerate(dataset):
        if has_gold_id:
            # Count from positive_ctxs or ground_truths
            if 'positive_ctxs' in item:
                gold_clusters = item['positive_ctxs']
            elif 'ground_truths' in item:
                gold_clusters = item['ground_truths']
            else:
                gold_docs_per_question.append(0)
                questions_with_no_golds += 1
                continue
            
            # Count unique gold doc IDs for this question
            gold_ids = set()
            for cluster in gold_clusters:
                for doc in cluster:
                    doc_id = doc.get('id', '')
                    if doc_id:
                        gold_ids.add(doc_id)
            
            num_golds = len(gold_ids)
        else:
            # For answer-based matching, we count the number of answer variants
            # Each answer can potentially match multiple documents, but we count
            # the number of distinct answers as a proxy for expected gold docs
            if 'answer_list' in item:
                # answer_list typically has format: [{'answer': '...', 'aliases': [...]}]
                num_golds = len(item['answer_list'])
            elif 'answers' in item:
                # answers typically has format: ['answer1', 'answer2', ...]
                num_golds = len(item['answers'])
            else:
                num_golds = 0
                questions_with_no_golds += 1
        
        total_gold_docs += num_golds
        gold_docs_per_question.append(num_golds)
    
    # Calculate statistics
    avg_gold_per_question = total_gold_docs / len(dataset) if dataset else 0
    median_gold = sorted(gold_docs_per_question)[len(gold_docs_per_question) // 2] if gold_docs_per_question else 0
    
    # Print results
    print("="*80)
    print("GROUND TRUTH GOLD DOCUMENT STATISTICS")
    print("="*80)
    print(f"\nGround truth file: {ground_truth_path}")
    print(f"Has gold IDs: {has_gold_id}")
    print(f"\nTotal questions: {len(dataset)}")
    print(f"Questions with no gold docs: {questions_with_no_golds}")
    print(f"\nTotal gold documents: {total_gold_docs}")
    print(f"Average gold docs per question: {avg_gold_per_question:.2f}")
    print(f"Median gold docs per question: {median_gold}")
    print(f"Min gold docs per question: {min(gold_docs_per_question) if gold_docs_per_question else 0}")
    print(f"Max gold docs per question: {max(gold_docs_per_question) if gold_docs_per_question else 0}")
    
    # Show distribution
    print(f"\nDistribution of gold docs per question:")
    from collections import Counter
    distribution = Counter(gold_docs_per_question)
    for count in sorted(distribution.keys())[:10]:  # Show first 10 bins
        num_questions = distribution[count]
        percentage = num_questions / len(dataset) * 100
        print(f"  {count} gold docs: {num_questions} questions ({percentage:.1f}%)")
    
    if len(distribution) > 10:
        print(f"  ... ({len(distribution) - 10} more bins)")
    
    print("\n" + "="*80)
    
    return total_gold_docs, avg_gold_per_question


def main():
    parser = argparse.ArgumentParser(
        description="Count total gold documents in ground truth dataset"
    )
    
    parser.add_argument(
        "--ground_truth_file",
        required=True,
        help="Path to ground truth JSONL file"
    )
    
    parser.add_argument(
        "--has_gold_id",
        action="store_true",
        help="Whether ground truth has gold document IDs (default: False, uses answer matching)"
    )
    
    args = parser.parse_args()
    
    total_golds, avg_golds = count_total_gold_documents(
        args.ground_truth_file,
        args.has_gold_id
    )
    
    print(f"\nSummary:")
    print(f"  Total gold documents: {total_golds}")
    print(f"  Average per question: {avg_golds:.2f}")
    print(f"\nTheoretical max recall@100: {min(100, avg_golds) / avg_golds * 100:.2f}%")
    print(f"  (Assumes you can retrieve {min(100, int(avg_golds))} gold docs per question)")

    zero_queries = find_zero_answer_queries(args.ground_truth_file)
    
    print(f"\n{'='*80}")
    print(f"Found {len(zero_queries)} queries with zero answers")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()