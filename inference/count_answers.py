#!/usr/bin/env python
"""
Count unique answers covered per round for histogram plotting.
Outputs simple statistics suitable for creating histograms.
"""

import json
import regex
import unicodedata
from typing import Dict, List
from tqdm import tqdm


class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens


def _normalize(text):
    return unicodedata.normalize('NFD', text)


def has_answer(answers, text, tokenizer) -> bool:
    """Check if a document contains an answer string."""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False


def read_jsonl(file_path):
    """Read JSONL file."""
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]


def read_json_file(file_path):
    """Read JSON or JSONL file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read().strip()
    
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        data = []
        for line in content.split('\n'):
            line = line.strip()
            if line:
                data.append(json.loads(line))
        return data


def count_answers_covered(retrieved_docs_path: str, ground_truth_path: str) -> Dict:
    """
    Count how many unique answers are covered by retrieved documents.
    Returns per-question statistics suitable for histogram plotting.
    """
    dataset = read_jsonl(ground_truth_path)
    retrieved_docs = read_json_file(retrieved_docs_path)
    tok = SimpleTokenizer()
    
    assert len(dataset) == len(retrieved_docs), \
        f'Length mismatch: {len(dataset)} vs {len(retrieved_docs)}'
    
    answers_covered_per_question = []  # List of counts for histogram
    
    total_answers_in_gt = 0
    total_answers_covered = 0
    
    for idx, (gold_inst, docs) in enumerate(tqdm(zip(dataset, retrieved_docs), 
                                                   desc="Counting answers covered",
                                                   total=len(dataset))):
        
        # Get answer list
        if 'answer_list' in gold_inst:
            answer_list_string = 'answer_list'
            answers = gold_inst['answer_list']
        else:
            answer_list_string = 'answers'
            answers = gold_inst['answers']
        
        # Count answers covered for this question
        answers_covered = set()
        
        for answer in answers:
            if answer_list_string == 'answer_list':
                answer_aliases = answer['aliases']
                answer_key = answer['answer_text']
            else:
                answer_aliases = [answer]
                answer_key = answer
            
            # Check if any document contains this answer
            for doc in docs.get('ctxs', []):
                if has_answer(answer_aliases, doc['text'], tok):
                    answers_covered.add(answer_key)
                    break  # Found this answer, move to next
        
        num_answers_total = len(answers)
        num_answers_covered = len(answers_covered)
        
        total_answers_in_gt += num_answers_total
        total_answers_covered += num_answers_covered
        answers_covered_per_question.append(num_answers_covered)
    
    # Calculate statistics
    avg_answers_covered = sum(answers_covered_per_question) / len(answers_covered_per_question)
    recall = total_answers_covered / total_answers_in_gt if total_answers_in_gt > 0 else 0
    
    return {
        'total_questions': len(dataset),
        'total_answers_in_ground_truth': total_answers_in_gt,
        'total_answers_covered': total_answers_covered,
        'average_answers_covered_per_question': avg_answers_covered,
        'recall': recall,
        'answers_covered_per_question': answers_covered_per_question  # For histogram
    }


def compare_rounds_for_histogram(round_files: List[str], 
                                 ground_truth_path: str,
                                 output_csv: str = None) -> None:
    """
    Compare answer coverage across rounds and output data for histogram.
    
    Args:
        round_files: List of paths to retrieved docs files (in order)
        ground_truth_path: Path to ground truth file
        output_csv: Optional path to save CSV for plotting
    """
    
    all_results = []
    
    print("\nProcessing rounds...")
    for round_num, file_path in enumerate(round_files, 1):
        print(f"\nRound {round_num}: {file_path}")
        result = count_answers_covered(file_path, ground_truth_path)
        result['round'] = round_num
        result['file'] = file_path
        all_results.append(result)
        
        print(f"  Total answers covered: {result['total_answers_covered']}")
        print(f"  Average per question: {result['average_answers_covered_per_question']:.2f}")
        print(f"  Recall: {result['recall']:.2%}")
    
    # Print comparison table
    print("\n" + "="*80)
    print("SUMMARY: Answers Covered Per Round")
    print("="*80)
    print(f"{'Round':<10} {'Total Answers':<18} {'Avg/Question':<15} {'Recall':<10}")
    print("-"*80)
    
    for result in all_results:
        print(f"{result['round']:<10} "
              f"{result['total_answers_covered']:<18} "
              f"{result['average_answers_covered_per_question']:<15.2f} "
              f"{result['recall']:<10.2%}")
    
    # Show improvements
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("INCREMENTAL IMPROVEMENTS")
        print("="*80)
        print(f"{'Round':<15} {'New Answers':<18} {'Recall Gain':<15}")
        print("-"*50)
        
        for i in range(1, len(all_results)):
            prev = all_results[i-1]
            curr = all_results[i]
            new_answers = curr['total_answers_covered'] - prev['total_answers_covered']
            recall_gain = curr['recall'] - prev['recall']
            
            print(f"R{prev['round']} -> R{curr['round']:<5} "
                  f"{new_answers:<18} "
                  f"{recall_gain:<15.2%}")
    
    # Save CSV for histogram plotting
    if output_csv:
        import csv
        
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['question_id'] + [f'round_{r["round"]}' for r in all_results])
            
            # Write data (one row per question)
            num_questions = all_results[0]['total_questions']
            for q_idx in range(num_questions):
                row = [q_idx]
                for result in all_results:
                    row.append(result['answers_covered_per_question'][q_idx])
                writer.writerow(row)
        
        print(f"\nHistogram data saved to: {output_csv}")
        print("Each row is a question, columns are rounds")
    
    # Also save summary JSON
    if output_csv:
        json_path = output_csv.replace('.csv', '_summary.json')
        with open(json_path, 'w') as f:
            json.dump({
                'rounds': [r['round'] for r in all_results],
                'total_answers_covered': [r['total_answers_covered'] for r in all_results],
                'average_per_question': [r['average_answers_covered_per_question'] for r in all_results],
                'recall': [r['recall'] for r in all_results],
                'per_question_data': {
                    f'round_{r["round"]}': r['answers_covered_per_question'] 
                    for r in all_results
                }
            }, f, indent=2)
        print(f"Summary JSON saved to: {json_path}")


def main():
    """
    Main function to count answers covered across rounds for histogram.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Count unique answers covered per round for histogram plotting"
    )
    
    parser.add_argument(
        "--ground_truth_file",
        required=True,
        help="Path to ground truth JSONL file"
    )
    
    parser.add_argument(
        "--round_files",
        nargs='+',
        required=True,
        help="Paths to retrieved docs files (in order by round)"
    )
    
    parser.add_argument(
        "--output_csv",
        default="answers_covered_by_round.csv",
        help="Path to save CSV output for histogram plotting"
    )
    
    args = parser.parse_args()
    
    compare_rounds_for_histogram(
        args.round_files,
        args.ground_truth_file,
        args.output_csv
    )
    
    print("\n" + "="*80)
    print("Done! Use the CSV file to create histograms in your plotting tool.")
    print("="*80)


if __name__ == "__main__":
    main()