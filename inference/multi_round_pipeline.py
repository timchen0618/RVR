#!/usr/bin/env python
"""
Multi-Round Retrieval Pipeline - Simplified for SBATCH execution
"""

import json
import os
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import sys
sys.path.append('/scratch/dq2024/diverse_retriever')

# Import functions from your existing code
from inference.get_all_golds import extract_all_gold_documents
from inference.concat import concat_documents_to_question
from inference.merge_r1_r2 import merge_first_and_second_round
from eval.eval_qampari import eval_retrieve_docs, read_jsonl, write_jsonl


class MultiRoundRetrievalPipeline:
    """Orchestrates multi-round retrieval with evaluation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize pipeline with configuration."""
        self.config = config
        self.round_results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directories
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary output directories."""
        self.base_output_dir = Path(self.config['output_base_dir'])
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.dirs = {
            'gold_docs': self.base_output_dir / 'gold_documents',
            'concatenated': self.base_output_dir / 'concatenated_queries',
            'retrieved': self.base_output_dir / 'retrieved_docs',
            'merged': self.base_output_dir / 'merged_docs',
            'evaluation': self.base_output_dir / 'evaluation_results',
            'sbatch_scripts': self.base_output_dir / 'sbatch_scripts'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def run_retrieval_sbatch(self, input_file: str, round_num: int) -> str:
        """
        Run retrieval directly using the current environment.
        """
        output_file = self.dirs['retrieved'] / f"round{round_num}_retrieved.json"
        
        print(f"\n{'='*60}")
        print(f"Round {round_num}: Running retrieval")
        print(f"{'='*60}")
        
        # Build command directly
        cmd = [
            "python", self.config['retrieval_script'],
            "--model_name_or_path", self.config['model_path'],
            "--passages", self.config['passages_file'],
            "--passages_embeddings", self.config['passages_embeddings'],
            "--data", str(input_file),
            "--output_dir", str(self.dirs['retrieved']),
            "--outfile", Path(output_file).name,
            "--training_mode", self.config['training_mode'],
            "--n_docs", str(self.config['n_docs_retrieve'])
        ]
        
        if self.config.get('norm_query', False):
            cmd.append("--norm_query")
        
        # Run directly in current environment - NO SBATCH NEEDED
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error in retrieval: {result.stderr}")
            raise RuntimeError(f"Retrieval failed for round {round_num}")
        
        print(f"Retrieval complete for round {round_num}")
        return str(output_file)
    
    def extract_gold_documents(self, retrieved_file: str, round_num: int) -> str:
        """Extract all gold documents from retrieved results."""
        output_file = self.dirs['gold_docs'] / f"round{round_num}_all_gold_docs.json"
        
        print(f"\n{'='*60}")
        print(f"Round {round_num}: Extracting gold documents")
        print(f"{'='*60}")
        
        stats = extract_all_gold_documents(
            retrieved_docs_path=retrieved_file,
            ground_truth_path=self.config['ground_truth_file'],
            output_path=str(output_file),
            has_gold_id=self.config['has_gold_id'],
            initial_topk=self.config['n_docs_retrieve']
        )
        
        return str(output_file)
    
    def read_json_file(self, file_path):
        
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
        
        # Try parsing as regular JSON first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # If that fails, try parsing as JSONL
            data = []
            for line in content.split('\n'):
                line = line.strip()
                if line:
                    data.append(json.loads(line))
            return data
    
    def concatenate_documents_to_queries(self, input_file: str, round_num: int, top_k: int) -> str:
        """Concatenate top-k documents to queries for next round."""
        # Load and filter to top-k documents
        data = self.read_json_file(input_file)
        
        # Extract original question and prepare for new concatenation
        for item in data:
            # Extract original question from concatenated format
            if 'question' in item and '\n\nDocuments:' in item['question']:
                # Split to get just the original question part
                original_question = item['question'].split('\n\nDocuments:')[0]
                # Remove "Question: " prefix if it exists
                if original_question.startswith('Question: '):
                    original_question = original_question[10:]
                item['question'] = original_question
                item['input'] = original_question
            
            # Keep only top-k documents for concatenation
            if 'ctxs' in item and len(item['ctxs']) > top_k:
                item['ctxs'] = item['ctxs'][:top_k]
        
        # Save filtered data with clean questions
        temp_file = self.dirs['concatenated'] / f"round{round_num}_top{top_k}_filtered.json"
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Concatenate documents to queries
        output_file = self.dirs['concatenated'] / f"round{round_num}_top{top_k}_concatenated.json"
        
        print(f"\n{'='*60}")
        print(f"Round {round_num}: Concatenating top-{top_k} documents to queries")
        print(f"{'='*60}")
        
        concat_documents_to_question(str(temp_file), str(output_file))
        
        return str(output_file)
    
    def merge_gold_with_retrieved(self, gold_file: str, retrieved_file: str, round_num: int) -> str:
        """Merge gold documents from previous rounds with new retrieved documents."""
        output_file = self.dirs['merged'] / f"round{round_num}_merged_100docs.json"
        
        print(f"\n{'='*60}")
        print(f"Round {round_num}: Merging gold docs with retrieved docs")
        print(f"{'='*60}")
        
        stats = merge_first_and_second_round(
            first_round_gold_path=gold_file,
            second_round_path=retrieved_file,
            output_path=str(output_file),
            target_total_docs=100
        )
        
        return str(output_file)

    def evaluate_file(self, docs_file: str, round_num: int, description: str = "") -> Dict[str, Any]:
        """Evaluate retrieval results."""
        print(f"\n{'='*60}")
        print(f"Round {round_num}: Evaluating {description}")
        print(f"{'='*60}")
        
        # Run evaluation
        scores = eval_retrieve_docs(
            docs_file,
            self.config['ground_truth_file'],
            has_gold_id=self.config['has_gold_id'],
            topk=100
        )
        
        # Parse scores based on has_gold_id
        if self.config['has_gold_id']:
            metrics = {
                'round': round_num,
                'description': description,
                'mrecall': float(scores[0]),
                'recall': float(scores[1]),
                'precision': float(scores[2]),
                'map': float(scores[3]),
                'ndcg': float(scores[4]),
                'mrr': float(scores[5])
            }
        else:
            metrics = {
                'round': round_num,
                'description': description,
                'mrecall': float(scores[0]),
                'recall': float(scores[1]),
                'precision': float(scores[2]),
                'mrr': float(scores[3])
            }
        
        return metrics
    
    def run_pipeline(self):
        """Execute the complete multi-round retrieval pipeline."""
        
        print(f"\n{'#'*80}")
        print(f"Starting Multi-Round Retrieval Pipeline")
        print(f"Starting from: {self.config['initial_merged_file']}")
        print(f"Total rounds: {self.config['total_rounds']}")
        print(f"Top-K concat: {self.config['top_k_concat']}")
        print(f"Output: {self.base_output_dir}")
        print(f"{'#'*80}\n")
        
        # Evaluate the initial 2-round file
        print(f"\n{'='*80}")
        print("Evaluating initial 2-round merged file")
        print(f"{'='*80}")
        
        initial_metrics = self.evaluate_file(
            self.config['initial_merged_file'],
            round_num=2,
            description="Initial 2-round merged results"
        )
        self.round_results.append(initial_metrics)
        
        # Extract gold documents from the 2-round file
        accumulated_gold = self.extract_gold_documents(
            self.config['initial_merged_file'],
            round_num=2
        )
        
        previous_merged = self.config['initial_merged_file']
        
        # Additional rounds (starting from round 3)
        for round_num in range(3, self.config['total_rounds'] + 1):
            print(f"\n{'='*80}")
            print(f"ROUND {round_num}: Iterative Retrieval")
            print(f"{'='*80}")
            
            # Concatenate top-k documents from previous merged results
            concatenated_file = self.concatenate_documents_to_queries(
                previous_merged,
                round_num=round_num,
                top_k=self.config['top_k_concat']
            )
            
            # Run retrieval with concatenated queries using SBATCH
            round_retrieved = self.run_retrieval_sbatch(
                concatenated_file,
                round_num=round_num
            )
            
            # Merge accumulated gold with new retrieved documents
            round_merged = self.merge_gold_with_retrieved(
                accumulated_gold,
                round_retrieved,
                round_num=round_num
            )
            
            # Evaluate merged results
            round_metrics = self.evaluate_file(
                round_merged,
                round_num=round_num,
                description=f"Round {round_num} merged results"
            )
            self.round_results.append(round_metrics)
            
            # Update for next round if needed
            if round_num < self.config['total_rounds']:
                accumulated_gold = self.extract_gold_documents(round_merged, round_num=round_num)
                previous_merged = round_merged
        
        # Save final results
        self.save_final_results()
        
        print(f"\n{'#'*80}")
        print("Pipeline Complete!")
        print(f"Results saved to: {self.base_output_dir}")
        print(f"{'#'*80}\n")
    
    def save_final_results(self):
        """Save evaluation results to files."""
        
        # Save results as JSON
        results_json = self.dirs['evaluation'] / f"all_rounds_results.json"
        with open(results_json, 'w') as f:
            json.dump(self.round_results, f, indent=2)
        
        # Save as readable text
        results_txt = self.dirs['evaluation'] / f"all_rounds_results.txt"
        with open(results_txt, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MULTI-ROUND RETRIEVAL RESULTS\n")
            f.write("="*80 + "\n\n")
            
            for metrics in self.round_results:
                f.write(f"Round {metrics['round']}: {metrics['description']}\n")
                f.write("-"*40 + "\n")
                f.write(f"  MRecall@100: {metrics['mrecall']:.2f}\n")
                f.write(f"  Recall@100:  {metrics['recall']:.2f}\n")
                f.write(f"  Precision@100: {metrics['precision']:.2f}\n")
                f.write(f"  MRR@100: {metrics['mrr']:.4f}\n")
                if 'ndcg' in metrics:
                    f.write(f"  nDCG@100: {metrics['ndcg']:.4f}\n")
                    f.write(f"  MAP@100: {metrics['map']:.4f}\n")
                f.write("\n")
        
        print(f"\nResults saved to:")
        print(f"  {results_json}")
        print(f"  {results_txt}")


def main():
    """Main entry point for the pipeline."""
    
    parser = argparse.ArgumentParser(description="Multi-Round Retrieval Pipeline")
    
    # Required arguments
    parser.add_argument("--initial_merged_file", required=True)
    parser.add_argument("--ground_truth_file", required=True)
    parser.add_argument("--passages_file", required=True)
    parser.add_argument("--passages_embeddings", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_base_dir", required=True)
    parser.add_argument("--retrieval_script", required=True)
    
    # Pipeline parameters
    parser.add_argument("--total_rounds", type=int, default=3)
    parser.add_argument("--top_k_concat", type=int, default=3, choices=[3, 6, 12])
    parser.add_argument("--n_docs_retrieve", type=int, default=100)
    parser.add_argument("--training_mode", default="base", choices=["base", "standard"])
    parser.add_argument("--has_gold_id", action="store_true")
    parser.add_argument("--norm_query", action="store_true")
    parser.add_argument("--conda_env", default="dr-env-3-9-qwen")
    
    args = parser.parse_args()
    config = vars(args)
    
    pipeline = MultiRoundRetrievalPipeline(config)
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()