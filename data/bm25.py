#!/usr/bin/env python
"""
BM25 Retrieval Script using existing BM25Retriever class
Compatible with dense retrieval pipeline format
"""

import os
import argparse
import json
import time
import csv
import sys
from pathlib import Path
from typing import List, Dict

# Set CSV field size limit
maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

# Add your project root to path if needed
sys.path.append('/scratch/dq2024/torch/diverse_retriever')

from src.retrieval.bm25_retriever import BM25Retriever


def load_data(data_path: str) -> List[Dict]:
    """Load query data from JSON or JSONL file"""
    try:
        if data_path.endswith(".json"):
            with open(data_path, "r") as f:
                data = json.load(f)
        elif data_path.endswith(".jsonl"):
            data = []
            with open(data_path, "r") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line.strip()))
        else:
            # Try as JSONL by default
            data = []
            with open(data_path, 'r') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line.strip()))
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
    
    return data


def load_passage_ids(passages_path: str) -> List[str]:
    """
    Load passage IDs from TSV file.
    TSV format: id \t text \t title
    """
    passage_ids = []
    print(f"Loading passage IDs from: {passages_path}", flush=True)
    
    with open(passages_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 1:
                passage_ids.append(row[0])
    
    print(f"Loaded {len(passage_ids)} passage IDs", flush=True)
    return passage_ids


class ImprovedBM25Retriever:
    """Wrapper around BM25Retriever that returns proper IDs and scores"""
    
    def __init__(self, corpus_path: str):
        print(f"Initializing BM25 retriever...", flush=True)
        print("This will tokenize the entire corpus, which may take a while...", flush=True)
        
        start_time = time.time()
        
        # Load passage IDs first
        self.passage_ids = load_passage_ids(corpus_path)
        
        # Initialize the old BM25Retriever
        self.retriever = BM25Retriever(corpus_path)
        
        print(f"Retriever initialized in {time.time() - start_time:.1f}s", flush=True)
    
    def retrieve(self, query: str, n: int) -> List[Dict]:
        """
        Retrieve documents and return them with proper IDs and scores.
        
        Returns:
            List of dicts with keys: id, title, text, score
        """
        from nltk import word_tokenize
        
        # Tokenize query
        tq = word_tokenize(query)
        
        # Get scores for all documents using BM25
        scores = self.retriever.bm25.get_scores(tq)
        
        # Get top-n indices
        top_indices = scores.argsort()[-n:][::-1]
        
        # Build results
        results = []
        for idx in top_indices:
            # Get the original corpus entry (format: "title[SEP]text")
            corpus_entry = self.retriever.corpus[idx]
            
            # Split to get title and text
            parts = corpus_entry.split('[SEP]')
            title = parts[0]
            text = parts[1] if len(parts) > 1 else ""
            
            # Get passage ID
            passage_id = self.passage_ids[idx]
            
            # Get score
            score = float(scores[idx])
            
            results.append({
                'id': passage_id,
                'title': title,
                'text': text,
                'score': str(score)
            })
        
        return results


def add_passages_to_data(data: List[Dict], retriever: ImprovedBM25Retriever, n_docs: int, question_key: str):
    """Add retrieved passages to query data"""
    print(f"Retrieving documents for {len(data)} queries...", flush=True)
    start_time = time.time()
    
    for i, query_data in enumerate(data):
        if i % 100 == 0:
            print(f"Processed {i}/{len(data)} queries", flush=True)
        
        # Get query
        query = query_data[question_key]
        
        # Retrieve documents
        docs = retriever.retrieve(query, n_docs)
        
        # Add to data
        query_data['ctxs'] = docs
    
    print(f"Retrieval completed in {time.time() - start_time:.1f}s", flush=True)


def main(args):
    # Initialize improved BM25 retriever
    retriever = ImprovedBM25Retriever(args.passages)
    
    # Load query data
    print(f"Loading queries from: {args.data}", flush=True)
    data = load_data(args.data)
    print(f"Loaded {len(data)} queries", flush=True)
    
    # Handle different training modes (for compatibility)
    if args.training_mode != "base":
        print(f"Note: BM25 does not support training_mode='{args.training_mode}'. Using base retrieval.", flush=True)
    
    # Retrieve documents
    add_passages_to_data(data, retriever, args.n_docs, args.question_key)
    
    # Save results
    output_path = os.path.join(args.output_dir, args.outfile)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Saving results to: {output_path}", flush=True)
    with open(output_path, "w") as f:
        for ex in data:
            json.dump(ex, f, ensure_ascii=False)
            f.write("\n")
    
    print(f"Results saved successfully", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BM25 Retrieval using existing BM25Retriever class")
    
    # Data arguments
    parser.add_argument("--data", required=True, type=str,
                       help="Path to queries (.json or .jsonl file)")
    parser.add_argument("--passages", type=str, required=True,
                       help="Path to passages (.tsv file)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for results")
    parser.add_argument("--outfile", type=str, default="bm25_output.jsonl",
                       help="Output filename")
    
    # Retrieval arguments
    parser.add_argument("--n_docs", type=int, default=100,
                       help="Number of documents to retrieve per query")
    parser.add_argument("--question_key", type=str, default="question",
                       help="Key for question field in data")
    
    # Compatibility arguments
    parser.add_argument("--training_mode", type=str, default="base",
                       help="Training mode (for compatibility - BM25 only supports base)")
    
    args = parser.parse_args()
    main(args)