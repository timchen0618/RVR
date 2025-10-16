import json
import numpy as np
import regex
import string
import unicodedata
from typing import List, Dict, Set
from collections import defaultdict

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

class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
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

def read_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

class DiversityScorer:
    """
    Formula: Diversity Score = (Unique Answers Found) / (Total Possible Answers)
    """
    
    def __init__(self):
        self.tokenizer = SimpleTokenizer()
    
    def calculate_diversity(self, results_path: str, ground_truth: Dict) -> float:
        # Load results from JSONL file
        results = read_jsonl(results_path)
        
        print(f"Loaded {len(results)} results from {results_path}")
        
        # Convert ground truth dict to list (sorted by keys to maintain order)
        ground_truth_list = [ground_truth[key] for key in sorted(ground_truth.keys())]
        
        coverage_scores = []
        matched_queries = 0
        
        # Match by position since IDs don't match
        for idx, (item, gt_data) in enumerate(zip(results, ground_truth_list)):
            if idx >= len(ground_truth_list):
                break
                
            matched_queries += 1
            
            if 'answer_list' in gt_data:
                answers = gt_data['answer_list']
                answer_list_key = 'answer_list'
            elif 'answers' in gt_data:
                answers = gt_data['answers']
                answer_list_key = 'answers'
            else:
                continue
            
            docs = item.get('ctxs', [])[:100]
            
            covered_answers = set()
            
            for doc in docs:
                doc_text = doc.get('text', '')
                
                for i, answer in enumerate(answers):
                    if answer_list_key == 'answer_list':
                        # Use the proper has_answer function with tokenizer
                        answer_aliases = answer.get('aliases', [answer.get('answer_text', '')])
                        if has_answer(answer_aliases, doc_text, self.tokenizer):
                            answer_id = answer.get('answer_text', str(i))
                            covered_answers.add(answer_id)
                    else:
                        # For simple answer lists
                        if has_answer([answer], doc_text, self.tokenizer):
                            covered_answers.add(answer)
            
            total_answers = len(answers)
            coverage = len(covered_answers) / total_answers if total_answers > 0 else 0
            coverage_scores.append(coverage)
            
            # Debug: print first few coverage scores
            if len(coverage_scores) <= 3:
                print(f"Query {idx}: {len(covered_answers)}/{total_answers} answers found, coverage={coverage:.3f}")
        
        print(f"Matched {matched_queries} queries")
        print(f"Calculated {len(coverage_scores)} coverage scores")
        mean_coverage = np.mean(coverage_scores) if coverage_scores else 0.0
        print(f"Mean coverage: {mean_coverage:.4f}")
        
        return mean_coverage
    
    def compare(self, json1_path: str, json2_path: str, ground_truth_path: str):
        # Load ground truth from JSONL file
        ground_truth = {}
        gt_list = read_jsonl(ground_truth_path)
        
        # Use index as key for sorted order
        for idx, item in enumerate(gt_list):
            ground_truth[idx] = item
        
        print(f"Loaded {len(ground_truth)} queries from ground truth")
        
        print("\nProcessing System 1:")
        score1 = self.calculate_diversity(json1_path, ground_truth)
        
        print("\nProcessing System 2:")
        score2 = self.calculate_diversity(json2_path, ground_truth)
        
        print(f"\nDiversity Scores (Answer Coverage):")
        print(f"System 1: {score1:.4f}")
        print(f"System 2: {score2:.4f}")
        
        if abs(score1 - score2) < 0.01:
            print("Result: TIE")
        elif score1 > score2:
            print(f"Result: System 1 wins by {(score1-score2):.4f}")
        else:
            print(f"Result: System 2 wins by {(score2-score1):.4f}")
        
        return score1, score2


if __name__ == "__main__":
    scorer = DiversityScorer()
    
    # score1, score2 = scorer.compare(
    #     #'/scratch/dq2024/diverse_retriever/eval/contriever/base/qampari_contreiver_tuned_base_model_0-3_input_test.json',
    #     '/scratch/dq2024/diverse_retriever/eval/contriever/base/qampari_corpus_base_finetuned_only_random_gold_inp0_AR_t200.json',
    #     '/scratch/dq2024/diverse_retriever/inference/contriever/standard/qampari_corpus_only_random_gold_AR_r1contr_r2std_0123_100.json', 
    #     '/scratch/dq2024/diverse_retriever/data/dev_data_gt_qampari_corpus.jsonl'
    # )
    score1, score2 = scorer.compare(
        #'/scratch/dq2024/diverse_retriever/eval/contriever/base/qampari_contreiver_tuned_base_model_0-3_input_test.json',
        '/scratch/dq2024/diverse_retriever/eval/qwen3/base/qampari_qwen3_finetuned_base_0-3_test.json',
        '/scratch/dq2024/diverse_retriever/inference/qwen3/standard/qampari_corpus_base_r1qw_r2std_AR_0123_t100.json', 
        '/scratch/dq2024/diverse_retriever/data/dev_data_gt_qampari_corpus.jsonl'
    )

    score1, score2 = scorer.compare(
        #'/scratch/dq2024/diverse_retriever/eval/contriever/base/qampari_contreiver_tuned_base_model_0-3_input_test.json',
        '/scratch/dq2024/diverse_retriever/eval/qwen3/base/qampari_qwen3_finetuned_base_0-6_test.json',
        '/scratch/dq2024/diverse_retriever/inference/qwen3/standard/qampari_corpus_base_r1qw_r2std_AR_0-6_t100.json', 
        '/scratch/dq2024/diverse_retriever/data/dev_data_gt_qampari_corpus.jsonl'
    )

    score1, score2 = scorer.compare(
        #'/scratch/dq2024/diverse_retriever/eval/contriever/base/qampari_contreiver_tuned_base_model_0-3_input_test.json',
        '/scratch/dq2024/diverse_retriever/eval/qwen3/base/qampari_qwen3_finetuned_base_0-12_test.json',
        '/scratch/dq2024/diverse_retriever/inference/qwen3/standard/qampari_corpus_base_r1qw_r2std_AR_0-12_t200.json', 
        '/scratch/dq2024/diverse_retriever/data/dev_data_gt_qampari_corpus.jsonl'
    )