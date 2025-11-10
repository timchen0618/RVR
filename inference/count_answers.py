#!/usr/bin/env python
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
    
    answers_covered_per_question = []
    preds = []  # Same structure as eval script
    
    total_answers_in_gt = 0
    
    for gold_inst, docs in tqdm(zip(dataset, retrieved_docs), 
                                desc="Counting answers covered",
                                total=len(dataset)):
        
        # Same logic as eval script
        if 'answer_list' in gold_inst:
            answer_list_string = 'answer_list'
        else:
            answer_list_string = 'answers'
        
        pred_inst = []
        
        for answer_idx, answer in enumerate(gold_inst[answer_list_string]):
            pred_inst.append([])
            for rank, doc in enumerate(docs['ctxs'][:100]):
                if answer_list_string == 'answer_list':
                    pred = has_answer(answer['aliases'], doc['text'], tok)
                else:
                    pred = has_answer([answer], doc['text'], tok)
                pred_inst[-1].append(pred)
        
        preds.append(pred_inst)
        
        # Count answers covered for this question
        num_answers_covered = sum([any(preds_per_answer) for preds_per_answer in pred_inst])
        answers_covered_per_question.append(num_answers_covered)
        total_answers_in_gt += len(gold_inst[answer_list_string])
    
    # Use exact same recall calculation as eval script
    total_answers_covered = sum(answers_covered_per_question)
    avg_answers_covered = sum(answers_covered_per_question) / len(answers_covered_per_question)
    
    # Macro recall - same as score_recall() in eval script
    recalls = []
    for inst in preds:
        recalls.append(sum([any(preds_per_perspective) for preds_per_perspective in inst])/len(inst))
    macro_recall = sum(recalls) / float(len(recalls))
    
    micro_recall = total_answers_covered / total_answers_in_gt if total_answers_in_gt > 0 else 0
    
    return {
        'total_questions': len(dataset),
        'total_answers_in_ground_truth': total_answers_in_gt,
        'total_answers_covered': total_answers_covered,
        'average_answers_covered_per_question': avg_answers_covered,
        'macro_recall': macro_recall,
        'micro_recall': micro_recall,
        'answers_covered_per_question': answers_covered_per_question
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
    
    print("\nProcessing rounds...",flush=True)
    for round_num, file_path in enumerate(round_files, 1):
        print(f"\nRound {round_num}: {file_path}",flush=True)
        result = count_answers_covered(file_path, ground_truth_path)
        result['round'] = round_num
        result['file'] = file_path
        all_results.append(result)
        
        print(f"  Total answers covered: {result['total_answers_covered']}",flush=True)
        print(f"  Average per question: {result['average_answers_covered_per_question']:.2f}",flush=True)
        print(f"  Macro Recall: {result['macro_recall']:.2%}",flush=True)
        print(f"  Micro Recall: {result['micro_recall']:.2%}",flush=True)
    
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
    print("GROUND TRUTH",flush=True)
    print("="*80)
    print(f"Total answers in ground truth: {total_answers_in_gt}",flush=True)
    print(f"Average answers per question: {avg_answers_in_gt:.2f}",flush=True)
    
    # Print comparison table
    print("\n" + "="*80)

    print("NEW ANSWERS FOUND PER ROUND",flush=True)
    print("="*80)
    print(f"{'Round':<10} {'Total New':<18} {'Avg New/Question':<20}",flush=True)
    print("-"*50)
    
    for new_stat in new_answers_per_round:
        print(f"{new_stat['round']:<10} "
              f"{new_stat['total_new_answers']:<18} "
              f"{new_stat['average_new_per_question']:<20.2f}",flush=True)
    
    # Show cumulative improvements
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("CUMULATIVE IMPROVEMENTS",flush=True)
        print("="*80)
        print(f"{'Round':<15} {'Cumulative Answers':<20} {'Macro Recall':<15}",flush=True)
        print("-"*50)
        
        for i in range(len(all_results)):
            cumulative_new = sum(new_answers_per_round[j]['total_new_answers'] for j in range(i+1))
            
            print(f"After R{all_results[i]['round']:<5} "
                  f"{cumulative_new:<20} "
                  f"{all_results[i]['macro_recall']:<15.2%}",flush=True)
    
    


def main():
    

    ground_truth_file = "/scratch/dq2024/diverse_retriever/data/dev_data_gt_qampari_corpus.jsonl"

    round_files = [
        "/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/contriever/base/gold_documents/round1_all_gold_docs.json",
        "/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/contriever/base/merged_docs/round2_merged_100docs.json",
        "/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/contriever/base/merged_docs/round3_merged_100docs.json",
        "/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/contriever/base/merged_docs/round4_merged_100docs.json",
        "/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/contriever/base/merged_docs/round5_merged_100docs.json",
        "/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/contriever/base/merged_docs/round6_merged_100docs.json",
        "/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/contriever/base/merged_docs/round7_merged_100docs.json",
        "/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/contriever/base/merged_docs/round8_merged_100docs.json",
        "/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/contriever/base/merged_docs/round9_merged_100docs.json",
        "/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline/contriever/base/merged_docs/round10_merged_100docs.json"
    ]
    compare_rounds_for_histogram(round_files, ground_truth_file)
    
    ground_truth_file = "/scratch/dq2024/diverse_retriever/data/dev_data_gt_qampari_corpus.jsonl"
    
    base_path = "/scratch/dq2024/diverse_retriever/eval/multi_round_pipeline"
    
    models = ["infly", "qwen3"]
    max_settings = ["max3", "max6", "max12"]
    base_settings = ["base", ""]  # base or nothing
    
    all_configurations = {}
    
    for model in models:
        for max_setting in max_settings:
            for base_setting in base_settings:
                # Build the path components
                if base_setting:
                    path_prefix = f"{base_path}/{model}/{max_setting}/{base_setting}"
                    config_name = f"{model}_{max_setting}_{base_setting}"
                else:
                    path_prefix = f"{base_path}/{model}/{max_setting}"
                    config_name = f"{model}_{max_setting}"
                
                round_files = [
                    f"{path_prefix}/gold_documents/round1_all_gold_docs.json",
                    f"{path_prefix}/merged_docs/round2_merged_100docs.json",
                    f"{path_prefix}/merged_docs/round3_merged_100docs.json",
                    f"{path_prefix}/merged_docs/round4_merged_100docs.json",
                    f"{path_prefix}/merged_docs/round5_merged_100docs.json",
                    f"{path_prefix}/merged_docs/round6_merged_100docs.json",
                    f"{path_prefix}/merged_docs/round7_merged_100docs.json",
                    f"{path_prefix}/merged_docs/round8_merged_100docs.json",
                    f"{path_prefix}/merged_docs/round9_merged_100docs.json",
                    f"{path_prefix}/merged_docs/round10_merged_100docs.json"
                ]
                
                print("\n" + "="*80, flush=True)
                print(f"CONFIGURATION: {config_name}", flush=True)
                print("="*80, flush=True)
                
                # Store results for this configuration
                config_results = compare_rounds_for_histogram(round_files, ground_truth_file)
                all_configurations[config_name] = config_results
    
    # Write all results to JSON file
    output_file = "out_files/all_configurations_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_configurations, f, indent=2)

    print("\n" + "="*80, flush=True)
    print("ALL CONFIGURATIONS COMPLETE", flush=True)
    print(f"Results saved to: {output_file}", flush=True)
    print("="*80, flush=True)

if __name__ == "__main__":
    main()