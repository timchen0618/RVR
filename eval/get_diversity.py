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
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        coverage_scores = []
        
        for query_id, query_data in results.items():
            if query_id not in ground_truth:
                continue
            
            gt_data = ground_truth[query_id]
            answer_list_key = 'answer_list' if 'answer_list' in gt_data else 'answers'
            answers = gt_data[answer_list_key]
            
            # Get retrieved documents
            docs = query_data.get('documents', query_data.get('retrieved_docs', []))
            
            # Find which answers are covered
            covered_answers = set()
            
            for doc in docs:
                doc_text = doc.get('text', doc.get('content', ''))
                
                for i, answer in enumerate(answers):
                    # Handle different answer formats
                    if answer_list_key == 'answer_list':
                        answer_aliases = answer.get('aliases', [answer.get('answer_text', '')])
                        answer_id = answer.get('answer_text', str(i))
                    else:
                        answer_aliases = [answer]
                        answer_id = answer
                    
                    if self.has_answer(answer_aliases, doc_text):
                        covered_answers.add(answer_id)
            
            # Calculate coverage for this query
            total_answers = len(answers)
            coverage = len(covered_answers) / total_answers if total_answers > 0 else 0
            coverage_scores.append(coverage)
        
        # Return mean coverage across all queries
        return np.mean(coverage_scores) if coverage_scores else 0.0
    
    def compare(self, json1_path: str, json2_path: str, ground_truth_path: str):
        """Compare diversity between two retrieval systems."""
        
        # Load ground truth
        with open(ground_truth_path, 'r') as f:
            ground_truth = json.load(f)
        
        # Calculate scores
        score1 = self.calculate_diversity(json1_path, ground_truth)
        score2 = self.calculate_diversity(json2_path, ground_truth)
        
        # Print results
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


# Example usage
if __name__ == "__main__":
    scorer = DiversityScorer()
    
    # Compare two systems
    score1, score2 = scorer.compare(
         "/scratch/dq2024/diverse_retriever/eval/contriever/base/qampari_contreiver_tuned_base_model_0-3_input_test.json",
        '/scratch/dq2024/diverse_retriever/inference/contriever/standard/qampari_corpus_only_random_gold_AR_r1contr_r2std_0123_100.json', 
        '/scratch/dq2024/diverse_retriever/data/dev_data_gt_qampari_corpus.jsonl'
    )