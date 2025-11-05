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
    """Read JSONL file, handling empty lines and encoding issues."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # Skip malformed lines
    return data


def read_json_file(file_path):
    """Read JSON or JSONL file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read().strip()
    
    if not content:
        return []
    
    # Try as JSON array first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    # Try as JSONL
    data = []
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError:
            continue
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
    
    # Calculate macro-averaged recall (same as eval script)
    per_question_recalls = []
    for q_idx in range(len(dataset)):
        if 'answer_list' in dataset[q_idx]:
            num_answers = len(dataset[q_idx]['answer_list'])
        else:
            num_answers = len(dataset[q_idx]['answers'])
        
        if num_answers > 0:
            q_recall = answers_covered_per_question[q_idx] / num_answers
        else:
            q_recall = 0
        per_question_recalls.append(q_recall)
    
    macro_recall = sum(per_question_recalls) / len(per_question_recalls) if per_question_recalls else 0
    micro_recall = total_answers_covered / total_answers_in_gt if total_answers_in_gt > 0 else 0
    
    return {
        'total_questions': len(dataset),
        'total_answers_in_ground_truth': total_answers_in_gt,
        'total_answers_covered': total_answers_covered,
        'average_answers_covered_per_question': avg_answers_covered,
        'macro_recall': macro_recall,  # This matches your evaluation script
        'micro_recall': micro_recall,
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
        print(f"  Macro Recall: {result['macro_recall']:.2%}")
        print(f"  Micro Recall: {result['micro_recall']:.2%}")
    
    # Calculate NEW answers per round (per question)
    new_answers_per_round = []
    
    for i in range(len(all_results)):
        if i == 0:
            # First round - all answers are "new"
            new_answers_per_question = all_results[0]['answers_covered_per_question']
            total_new = all_results[0]['total_answers_covered']
            avg_new = all_results[0]['average_answers_covered_per_question']
        else:
            # Calculate new answers per question
            new_answers_per_question = []
            total_new = 0
            
            prev_answers = all_results[i-1]['answers_covered_per_question']
            curr_answers = all_results[i]['answers_covered_per_question']
            
            for prev_count, curr_count in zip(prev_answers, curr_answers):
                new_count = max(0, curr_count - prev_count)  # New answers found
                new_answers_per_question.append(new_count)
                total_new += new_count
            
            avg_new = sum(new_answers_per_question) / len(new_answers_per_question)
        
        new_answers_per_round.append({
            'round': all_results[i]['round'],
            'new_answers_per_question': new_answers_per_question,
            'total_new_answers': total_new,
            'average_new_per_question': avg_new
        })
    
    # Print ground truth info
    total_answers_in_gt = all_results[0]['total_answers_in_ground_truth']
    avg_answers_in_gt = total_answers_in_gt / all_results[0]['total_questions']
    
    print("\n" + "="*80)
    print("GROUND TRUTH")
    print("="*80)
    print(f"Total answers in ground truth: {total_answers_in_gt}")
    print(f"Average answers per question: {avg_answers_in_gt:.2f}")
    
    # Print comparison table
    print("\n" + "="*80)
    print("SUMMARY: Answers Covered Per Round")
    print("="*80)
    print(f"{'Round':<10} {'Total Answers':<18} {'Avg/Question':<15} {'Macro Recall':<15}")
    print("-"*80)
    
    for result in all_results:
        print(f"{result['round']:<10} "
              f"{result['total_answers_covered']:<18} "
              f"{result['average_answers_covered_per_question']:<15.2f} "
              f"{result['macro_recall']:<15.2%}")
    
    # Show NEW answers found per round
    print("\n" + "="*80)
    print("NEW ANSWERS FOUND PER ROUND")
    print("="*80)
    print(f"{'Round':<10} {'Total New':<18} {'Avg New/Question':<20}")
    print("-"*50)
    
    for new_stat in new_answers_per_round:
        print(f"{new_stat['round']:<10} "
              f"{new_stat['total_new_answers']:<18} "
              f"{new_stat['average_new_per_question']:<20.2f}")
    
    # Show cumulative improvements
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("CUMULATIVE IMPROVEMENTS")
        print("="*80)
        print(f"{'Round':<15} {'Cumulative Answers':<20} {'Macro Recall':<15}")
        print("-"*50)
        
        for i in range(len(all_results)):
            cumulative_new = sum(new_answers_per_round[j]['total_new_answers'] for j in range(i+1))
            
            print(f"After R{all_results[i]['round']:<5} "
                  f"{cumulative_new:<20} "
                  f"{all_results[i]['macro_recall']:<15.2%}")
    
    # Save CSV for histogram plotting
    if output_csv:
        import csv
        
        # Save cumulative answers covered
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['question_id'] + [f'round_{r["round"]}_cumulative' for r in all_results])
            
            # Write data (one row per question)
            num_questions = all_results[0]['total_questions']
            for q_idx in range(num_questions):
                row = [q_idx]
                for result in all_results:
                    row.append(result['answers_covered_per_question'][q_idx])
                writer.writerow(row)
        
        print(f"\nCumulative answers data saved to: {output_csv}")
        
        # Save NEW answers per round
        new_csv = output_csv.replace('.csv', '_new_answers.csv')
        with open(new_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['question_id'] + [f'round_{stat["round"]}_new' for stat in new_answers_per_round])
            
            # Write data (one row per question)
            num_questions = all_results[0]['total_questions']
            for q_idx in range(num_questions):
                row = [q_idx]
                for new_stat in new_answers_per_round:
                    row.append(new_stat['new_answers_per_question'][q_idx])
                writer.writerow(row)
        
        print(f"NEW answers per round saved to: {new_csv}")
        print("Each row is a question, columns show NEW answers found in that round")
    
    # Also save summary JSON
    if output_csv:
        json_path = output_csv.replace('.csv', '_summary.json')
        with open(json_path, 'w') as f:
            json.dump({
                'rounds': [r['round'] for r in all_results],
                'total_answers_in_ground_truth': all_results[0]['total_answers_in_ground_truth'],
                'total_answers_covered_cumulative': [r['total_answers_covered'] for r in all_results],
                'total_new_answers_per_round': [stat['total_new_answers'] for stat in new_answers_per_round],
                'average_per_question_cumulative': [r['average_answers_covered_per_question'] for r in all_results],
                'average_new_per_question': [stat['average_new_per_question'] for stat in new_answers_per_round],
                'macro_recall': [r['macro_recall'] for r in all_results],
                'micro_recall': [r['micro_recall'] for r in all_results],
                'per_question_cumulative': {
                    f'round_{r["round"]}': r['answers_covered_per_question'] 
                    for r in all_results
                },
                'per_question_new': {
                    f'round_{stat["round"]}': stat['new_answers_per_question']
                    for stat in new_answers_per_round
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


'''
python count_answers.py \
    --ground_truth_file /scratch/dq2024/torch/diverse_retriever/data/dev_data_gt_qampari_corpus.jsonl \
    --round_files \
        /scratch/dq2024/torch/diverse_retriever/eval/multi_round_pipeline/contriever/gold_documents/round1_all_gold_docs.json \
        /scratch/dq2024/torch/diverse_retriever/eval/multi_round_pipeline/contriever/merged_docs/round2_merged_100docs.json \
    --output_csv answers_covered_by_round.csv


'''