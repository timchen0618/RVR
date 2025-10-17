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
        
        Args:
            input_file: Path to input JSON file with queries
            round_num: Current round number
            
        Returns:
            Path to output file
        """
        output_file = self.dirs['retrieved'] / f"round{round_num}_retrieved.json"
        
        print(f"\n{'='*60}")
        print(f"Round {round_num}: Running retrieval")
        print(f"{'='*60}")
        
        # Build command
        norm_query_flag = "--norm_query" if self.config.get('norm_query', False) else ""
        
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
        
        print(f"Running retrieval command...")
        print(f"Output will be saved to: {output_file}")
        
        # Run directly in current environment
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
        """Read JSON or JSONL file."""
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
        # Load data - use read_json_file which handles both JSON and JSONL
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
    
    def merge_all_gold_with_retrieved(self, gold_files: List[str], retrieved_file: str, round_num: int) -> str:
        """
        Merge ALL gold documents from all previous rounds with new retrieved documents.
        
        Args:
            gold_files: List of paths to gold document files from all rounds
            retrieved_file: Path to newly retrieved documents
            round_num: Current round number
            
        Returns:
            Path to merged documents file
        """
        output_file = self.dirs['merged'] / f"round{round_num}_merged_100docs.json"
        
        print(f"\n{'='*60}")
        print(f"Round {round_num}: Merging ALL gold docs with retrieved docs")
        print(f"{'='*60}")
        
        # Load all gold documents from all rounds
        all_gold_docs = []
        gold_doc_ids = set()
        
        for gold_file in gold_files:
            gold_data = self.read_json_file(gold_file)
            for item in gold_data:
                if 'ctxs' in item:
                    for doc in item['ctxs']:
                        doc_id = doc.get('id', '')
                        if doc_id not in gold_doc_ids:
                            all_gold_docs.append(doc)
                            gold_doc_ids.add(doc_id)
        
        print(f"Total unique gold documents from all rounds: {len(all_gold_docs)}")
        
        # Load retrieved documents
        retrieved_data = self.read_json_file(retrieved_file)
        
        # Create merged output
        merged_data = []
        for idx, retrieved_item in enumerate(retrieved_data):
            merged_entry = {
                'question': retrieved_item.get('question', ''),
                'input': retrieved_item.get('input', ''),
                'id': retrieved_item.get('id', idx),
                'answers': retrieved_item.get('answers', []),
                'ctxs': []
            }
            
            # Start with ALL gold documents (sorted by score)
            question_gold_docs = sorted(all_gold_docs, key=lambda x: float(x.get('score', 0)), reverse=True)
            
            # Add gold docs up to limit
            final_docs = question_gold_docs[:100] if len(question_gold_docs) <= 100 else question_gold_docs[:100]
            
            # If we have less than 100 gold docs, fill with retrieved non-duplicates
            if len(final_docs) < 100:
                remaining_slots = 100 - len(final_docs)
                for doc in retrieved_item.get('ctxs', []):
                    if remaining_slots <= 0:
                        break
                    if doc.get('id', '') not in gold_doc_ids:
                        final_docs.append(doc)
                        remaining_slots -= 1
            
            merged_entry['ctxs'] = final_docs
            merged_data.append(merged_entry)
        
        # Save merged results
        write_jsonl(merged_data, str(output_file))
        
        print(f"Merged {len(merged_data)} questions")
        print(f"Results saved to: {output_file}")
        
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
        print(f"Initial retrieval file: {self.config['initial_retrieval_file']}")
        print(f"Total rounds: {self.config['total_rounds']}")
        print(f"Top-K concat: {self.config['top_k_concat']}")
        print(f"Output: {self.base_output_dir}")
        print(f"{'#'*80}\n")
        
        # Extract gold documents from the initial retrieval file
        print(f"\n{'='*80}")
        print("Extracting gold documents from initial retrieval")
        print(f"{'='*80}")
        
        all_accumulated_gold = self.extract_gold_documents(
            self.config['initial_retrieval_file'],
            round_num=1
        )
        
        # List to accumulate ALL gold document files from each round
        all_gold_files = [all_accumulated_gold]
        
        # For concatenation, we'll use the most recent gold documents
        recent_gold_for_concat = all_accumulated_gold
        
        # Run additional rounds
        for round_num in range(2, self.config['total_rounds'] + 1):
            print(f"\n{'='*80}")
            print(f"ROUND {round_num}: Iterative Retrieval")
            print(f"{'='*80}")
            
            # Concatenate top-k documents from recent gold documents
            concatenated_file = self.concatenate_documents_to_queries(
                recent_gold_for_concat,
                round_num=round_num,
                top_k=self.config['top_k_concat']
            )
            
            # Run retrieval with concatenated queries
            round_retrieved = self.run_retrieval_sbatch(
                concatenated_file,
                round_num=round_num
            )
            
            # Extract gold documents from the newly retrieved results
            round_gold = self.extract_gold_documents(
                round_retrieved,
                round_num=round_num
            )
            all_gold_files.append(round_gold)
            
            # Merge ALL accumulated gold documents with new retrieved documents
            if round_num == self.config['total_rounds']:
                # Final round: merge all gold docs from all rounds with retrieved
                print(f"\nFinal round: Merging all gold documents from rounds 1-{round_num}")
                round_merged = self.merge_all_gold_with_retrieved(
                    all_gold_files,
                    round_retrieved,
                    round_num=round_num
                )
            else:
                # Not final round: just save the gold for next iteration
                recent_gold_for_concat = round_gold
                print(f"Gold documents from round {round_num} will be used for round {round_num+1} concatenation")
                continue
            
            # Evaluate merged results (only for final round now)
            round_metrics = self.evaluate_file(
                round_merged,
                round_num=round_num,
                description=f"Final round {round_num} merged results"
            )
            self.round_results.append(round_metrics)
        
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
    parser.add_argument("--initial_retrieval_file", required=True,
                       help="Initial retrieval file (e.g., round 1 retrieved docs)")
    parser.add_argument("--ground_truth_file", required=True)
    parser.add_argument("--passages_file", required=True)
    parser.add_argument("--passages_embeddings", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_base_dir", required=True)
    parser.add_argument("--retrieval_script", required=True)
    
    # Pipeline parameters
    parser.add_argument("--total_rounds", type=int, default=3)
    parser.add_argument("--top_k_concat", type=int, default=3, choices=[3, 6, 12])
    parser.add_argument("--n_docs_retrieve", type=int, default=200)
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