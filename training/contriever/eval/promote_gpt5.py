#!/usr/bin/env python3
"""
GPT-5 Verifier Document Reranker

This script uses GPT-5 results as a verifier to rerank retrieved documents.
Documents that match GPT-5 "gold" documents are promoted to the top.
Output is limited to 100 documents per query.
"""

import json
import sys
from typing import Dict, List, Set
from collections import defaultdict


def load_json_file(filepath: str) -> List[Dict]:
    """Load JSON file and return list of objects."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f if line.strip()]
        return data
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {filepath}: {e}")
        sys.exit(1)


def save_json_file(data: List[Dict], filepath: str) -> None:
    """Save data to JSON file (one JSON object per line)."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Reranked results saved to: {filepath}")
    except Exception as e:
        print(f"Error saving file {filepath}: {e}")
        sys.exit(1)


def extract_document_ids(contexts: List[Dict]) -> Set[str]:
    """Extract document IDs from contexts list."""
    return {ctx['id'] for ctx in contexts}


def rerank_documents(retrieval_results: List[Dict], gpt5_verifier: List[Dict], max_docs: int = 100) -> tuple[List[Dict], Dict]:
    """
    Rerank documents by promoting GPT-5 verified documents to the top.
    Limit output to max_docs per query.
    
    Args:
        retrieval_results: List of queries with retrieved documents
        gpt5_verifier: List of queries with GPT-5 "gold" documents
        max_docs: Maximum number of documents to keep per query (default: 100)
    
    Returns:
        Tuple of (reranked results list, statistics dictionary)
    """
    # Create lookup dictionary for GPT-5 gold documents by query ID
    gpt5_dict = {item['id']: item for item in gpt5_verifier}
    
    reranked_results = []
    stats = {
        'total_queries': 0,
        'queries_with_matches': 0,
        'total_promoted_docs': 0,
        'avg_promoted_per_query': 0.0,
        'total_input_docs': 0,
        'total_output_docs': 0,
        'avg_input_docs_per_query': 0.0,
        'avg_output_docs_per_query': 0.0
    }
    
    print(f"Processing queries (limiting to {max_docs} docs per query)...")
    
    for query_item in retrieval_results:
        query_id = query_item['id']
        stats['total_queries'] += 1
        stats['total_input_docs'] += len(query_item['ctxs'])
        
        # Get GPT-5 gold document IDs for this query
        if query_id not in gpt5_dict:
            print(f"Warning: Query {query_id} not found in GPT-5 verifier file")
            # Still limit to max_docs even without GPT-5 data
            limited_contexts = query_item['ctxs'][:max_docs]
            reranked_query = query_item.copy()
            reranked_query['ctxs'] = limited_contexts
            stats['total_output_docs'] += len(limited_contexts)
            reranked_results.append(reranked_query)
            continue
        
        gpt5_gold_ids = extract_document_ids(gpt5_dict[query_id]['ctxs'])
        
        # Separate retrieved documents into promoted (matching GPT-5) and remaining
        promoted_docs = []
        remaining_docs = []
        seen_ids = set()
        
        for doc in query_item['ctxs']:
            doc_id = doc['id']
            # Skip duplicates
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            
            if doc_id in gpt5_gold_ids:
                promoted_docs.append(doc)
            else:
                remaining_docs.append(doc)
        
        # Update statistics
        if promoted_docs:
            stats['queries_with_matches'] += 1
            stats['total_promoted_docs'] += len(promoted_docs)
        
        # Create reranked context list: promoted docs first, then remaining docs
        # Limit total to max_docs
        reranked_contexts = promoted_docs[:max_docs]  # First, add promoted docs (up to max_docs)
        
        # Fill remaining slots with non-promoted documents
        remaining_slots = max_docs - len(reranked_contexts)
        if remaining_slots > 0 and remaining_docs:
            reranked_contexts.extend(remaining_docs[:remaining_slots])
        
        stats['total_output_docs'] += len(reranked_contexts)
        
        # Create new query item with reranked contexts
        reranked_query = query_item.copy()
        reranked_query['ctxs'] = reranked_contexts
        
        reranked_results.append(reranked_query)
        
        # Print progress every 50 queries
        if stats['total_queries'] % 50 == 0:
            print(f"Processed {stats['total_queries']} queries...")
    
    # Calculate final statistics
    if stats['total_queries'] > 0:
        stats['avg_promoted_per_query'] = stats['total_promoted_docs'] / stats['total_queries']
        stats['avg_input_docs_per_query'] = stats['total_input_docs'] / stats['total_queries']
        stats['avg_output_docs_per_query'] = stats['total_output_docs'] / stats['total_queries']
    
    return reranked_results, stats


def print_statistics(stats: Dict, sample_results: List[Dict], gpt5_dict: Dict, max_docs: int = 100) -> None:
    """Print detailed statistics about the reranking process."""
    print("\n" + "="*80)
    print("RERANKING STATISTICS")
    print("="*80)
    print(f"Total queries processed: {stats['total_queries']}")
    print(f"Queries with promoted documents: {stats['queries_with_matches']}")
    print(f"Queries with no matches: {stats['total_queries'] - stats['queries_with_matches']}")
    print(f"Total documents promoted: {stats['total_promoted_docs']}")
    print(f"Average promoted docs per query: {stats['avg_promoted_per_query']:.2f}")
    
    print(f"\nDocument counts:")
    print(f"  Total input documents: {stats['total_input_docs']}")
    print(f"  Total output documents: {stats['total_output_docs']}")
    print(f"  Average input docs per query: {stats['avg_input_docs_per_query']:.1f}")
    print(f"  Average output docs per query: {stats['avg_output_docs_per_query']:.1f}")
    print(f"  Target docs per query: {max_docs}")
    
    if stats['queries_with_matches'] > 0:
        avg_promoted_per_matching_query = stats['total_promoted_docs'] / stats['queries_with_matches']
        print(f"Average promoted docs per matching query: {avg_promoted_per_matching_query:.2f}")
    
    percentage_with_matches = (stats['queries_with_matches'] / stats['total_queries']) * 100
    print(f"Percentage of queries with matches: {percentage_with_matches:.1f}%")
    
    # Show examples of promoted documents
    print("\n" + "="*80)
    print("SAMPLE RERANKING EXAMPLES")
    print("="*80)
    
    examples_shown = 0
    for query_item in sample_results:
        if examples_shown >= 5:  # Show max 5 examples
            break
            
        query_id = query_item['id']
        if query_id not in gpt5_dict:
            continue
            
        gpt5_gold_ids = extract_document_ids(gpt5_dict[query_id]['ctxs'])
        
        # Count promoted documents (those at the beginning that match GPT-5)
        promoted_count = 0
        for doc in query_item['ctxs']:
            if doc['id'] in gpt5_gold_ids:
                promoted_count += 1
            else:
                break  # Stop when we hit the first non-promoted doc
        
        if promoted_count > 0:
            print(f"Query {query_id}: {promoted_count} documents promoted to top")
            print(f"  Question: {query_item.get('question', 'N/A')[:100]}...")
            print(f"  GPT-5 gold docs: {len(gpt5_gold_ids)}")
            print(f"  Output docs (limited): {len(query_item['ctxs'])}")
            print(f"  Promoted doc IDs: {[doc['id'] for doc in query_item['ctxs'][:promoted_count]]}")
            print()
            examples_shown += 1


def main():
    # Configuration
    MAX_DOCS_PER_QUERY = 100  # Output will have exactly this many docs per query
    
    # Hardcoded file paths
    retrieval_file = '/scratch/dq2024/diverse_retriever/training/contriever/eval/base/qampari_corpus_base_only_random_inp0_t200.json'
    gpt5_verifier_file = '/scratch/dq2024/diverse_retriever/training/contriever/eval/qwen3_verifier/qampari_base_qwen3-30B_all_gold_docs_200_vllm.json'
    output_file = '/scratch/dq2024/diverse_retriever/training/contriever/eval/base/qampari_corpus_base_only_random_inp0_t200_promote-qwen3.json' 
    
    print("Loading files...")
    retrieval_results = load_json_file(retrieval_file)
    gpt5_verifier = load_json_file(gpt5_verifier_file)
    
    print(f"Loaded {len(retrieval_results)} queries from retrieval results")
    print(f"Loaded {len(gpt5_verifier)} queries from GPT-5 verifier")
    
    # Create GPT-5 lookup dictionary
    gpt5_dict = {item['id']: item for item in gpt5_verifier}
    
    # Perform reranking with document limit
    reranked_results, stats = rerank_documents(retrieval_results, gpt5_verifier, max_docs=MAX_DOCS_PER_QUERY)
    
    # Print statistics
    print_statistics(stats, reranked_results, gpt5_dict, max_docs=MAX_DOCS_PER_QUERY)
    
    # Save results
    save_json_file(reranked_results, output_file)
    
    print(f"\nReranking complete!")
    print(f"Original queries: {len(retrieval_results)}")
    print(f"Reranked queries: {len(reranked_results)}")
    print(f"Documents per query in output: {MAX_DOCS_PER_QUERY}")


if __name__ == "__main__":
    main()