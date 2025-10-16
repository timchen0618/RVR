import json
import numpy as np
from typing import List, Dict, Set
from collections import defaultdict

class DiversityScorer:
    """
    Formula: Diversity Score = (Unique Answers Found) / (Total Possible Answers)
    """
    
    def __init__(self):
        pass
    
    def has_answer(self, answer_aliases: List[str], text: str) -> bool:
        text_lower = text.lower()
        for alias in answer_aliases:
            if alias.lower() in text_lower:
                return True
        return False
    
    def calculate_diversity(self, results_path: str, ground_truth: Dict) -> float:
        # Load results from JSONL file
        results = []
        with open(results_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        
        print(f"Loaded {len(results)} results from {results_path}")
        
        # Convert ground truth dict to list (sorted by keys to maintain order)
        ground_truth_list = [ground_truth[key] for key in sorted(ground_truth.keys())]
        
        coverage_scores = []
        matched_queries = 0
        
        # Match by position since IDs don't match
        for idx, item in enumerate(results):
            if idx >= len(ground_truth_list):
                break
                
            matched_queries += 1
            gt_data = ground_truth_list[idx]
            
            if 'answer_list' in gt_data:
                answers = gt_data['answer_list']
                answer_list_key = 'answer_list'
            elif 'answers' in gt_data:
                answers = gt_data['answers']
                answer_list_key = 'answers'
            else:
                continue
            
            docs = item.get('ctxs', [])
            
            covered_answers = set()
            
            for doc in docs:
                doc_text = doc.get('text', '')
                
                for i, answer in enumerate(answers):
                    if answer_list_key == 'answer_list':
                        answer_aliases = answer.get('aliases', [answer.get('answer_text', '')])
                        answer_id = answer.get('answer_text', str(i))
                    else:
                        answer_aliases = [answer] if isinstance(answer, str) else [str(answer)]
                        answer_id = answer if isinstance(answer, str) else str(answer)
                    
                    if self.has_answer(answer_aliases, doc_text):
                        covered_answers.add(answer_id)
            
            total_answers = len(answers)
            coverage = len(covered_answers) / total_answers if total_answers > 0 else 0
            coverage_scores.append(coverage)
            
            # Debug: print first few coverage scores
            if len(coverage_scores) <= 3:
                print(f"Query {idx}: {len(covered_answers)}/{total_answers} answers found, coverage={coverage:.3f}")
        
        print(f"Matched {matched_queries} queries")
        print(f"Calculated {len(coverage_scores)} coverage scores")
        
        return np.mean(coverage_scores) if coverage_scores else 0.0
    
    def compare(self, json1_path: str, json2_path: str, ground_truth_path: str):
        # Load ground truth from JSONL file
        ground_truth = {}
        with open(ground_truth_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    # Get the ID from the item
                    if 'qid' in item:
                        ground_truth[item['qid']] = item
                    elif 'id' in item:
                        ground_truth[item['id']] = item
                    elif 'question_id' in item:
                        ground_truth[item['question_id']] = item
        
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
    
    score1, score2 = scorer.compare(
        '/scratch/dq2024/diverse_retriever/eval/contriever/base/qampari_contreiver_tuned_base_model_0-3_input_test.json',
        '/scratch/dq2024/diverse_retriever/inference/contriever/standard/qampari_corpus_only_random_gold_AR_r1contr_r2std_0123_100.json', 
        '/scratch/dq2024/diverse_retriever/data/dev_data_gt_qampari_corpus.jsonl'
    )