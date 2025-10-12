#!/usr/bin/env python3
"""
GPT-5 Verifier Document Reranker

This script uses GPT-5 results as a verifier to rerank retrieved documents.
Documents that match GPT-5 "gold" documents are promoted to the top.
Output is limited to 100 documents per query.
"""

import json
import sys
from typing import Dict, List, Set, Tuple
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


def save_json_file(data: List[Dict], filepath: str, format_style: str = "jsonl") -> None:
    """
    Save data to JSON file.
    
    Args:
        data: List of dictionaries to save
        filepath: Output file path
        format_style: Either "jsonl" (one object per line) or "array" (formatted JSON array)
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            if format_style == "array":
                # Save as a formatted JSON array
                json.dump(data, f, ensure_ascii=False, indent=4)
            else:
                # Save as JSONL (one object per line)
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Results saved to: {filepath}")
    except Exception as e:
        print(f"Error saving file {filepath}: {e}")
        sys.exit(1)


def extract_document_ids(contexts: List[Dict]) -> Set[str]:
    """Extract document IDs from contexts list."""
    return {ctx['id'] for ctx in contexts}


def extract_input_negative_contexts(query_item: Dict, num_docs: int = 3) -> List[Dict]:
    """
    Extract top K highest scoring negative documents from ctxs.
    
    Args:
        query_item: Query dictionary containing ctxs
        num_docs: Number of documents to extract (default: 3)
    
    Returns:
        List of top K documents with highest scores
    """
    if 'ctxs' not in query_item or not query_item['ctxs']:
        return []
    
    # Sort documents by score in descending order
    sorted_docs = sorted(query_item['ctxs'], 
                        key=lambda x: float(x.get('score', 0)), 
                        reverse=True)
    
    # Return top K documents
    return sorted_docs[:num_docs]


def process_input_negative_mode(retrieval_results: List[Dict], num_negative_docs: int = 3) -> Tuple[List[Dict], Dict]:
    """
    Process queries to extract input_negative_ctxs from existing ctxs.
    
    Args:
        retrieval_results: List of queries with retrieved documents
        num_negative_docs: Number of negative documents to extract per query
    
    Returns:
        Tuple of (processed results list, statistics dictionary)
    """
    processed_results = []
    stats = {
        'total_queries': 0,
        'queries_with_negative_docs': 0,
        'total_negative_docs': 0,
        'avg_negative_docs_per_query': 0.0
    }
    
    print(f"Processing queries for input_negative_ctxs (extracting top {num_negative_docs} docs)...")
    
    for query_item in retrieval_results:
        stats['total_queries'] += 1
        
        # Extract top K negative contexts
        negative_contexts = extract_input_negative_contexts(query_item, num_negative_docs)
        
        if negative_contexts:
            stats['queries_with_negative_docs'] += 1
            stats['total_negative_docs'] += len(negative_contexts)
        
        # Create new query item with input_negative_ctxs
        processed_query = {
            'question': query_item.get('question', ''),
            'input': query_item.get('input', query_item.get('question', '')),
            'id': query_item.get('id', 0),
            'answers': query_item.get('answers', ['']),
            'ctxs': [],  # Empty ctxs as requested
            'input_negative_ctxs': negative_contexts
        }
        
        processed_results.append(processed_query)
        
        # Print progress every 50 queries
        if stats['total_queries'] % 50 == 0:
            print(f"Processed {stats['total_queries']} queries...")
    
    # Calculate final statistics
    if stats['total_queries'] > 0:
        stats['avg_negative_docs_per_query'] = stats['total_negative_docs'] / stats['total_queries']
    
    return processed_results, stats


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


def print_statistics(stats: Dict, sample_results: List[Dict], gpt5_dict: Dict = None, max_docs: int = 100, mode: str = "all") -> None:
    """Print detailed statistics about the reranking process."""
    
    if mode == "input":
        print("\n" + "="*80)
        print("INPUT NEGATIVE CONTEXTS EXTRACTION STATISTICS")
        print("="*80)
        print(f"Total queries processed: {stats['total_queries']}")
        print(f"Queries with negative documents: {stats['queries_with_negative_docs']}")
        print(f"Total negative documents extracted: {stats['total_negative_docs']}")
        print(f"Average negative docs per query: {stats['avg_negative_docs_per_query']:.2f}")
        
        if stats['queries_with_negative_docs'] > 0:
            avg_neg_per_matching_query = stats['total_negative_docs'] / stats['queries_with_negative_docs']
            print(f"Average negative docs per query with docs: {avg_neg_per_matching_query:.2f}")
        
        percentage_with_neg = (stats['queries_with_negative_docs'] / stats['total_queries']) * 100 if stats['total_queries'] > 0 else 0
        print(f"Percentage of queries with negative docs: {percentage_with_neg:.1f}%")
        
        # Show examples
        print("\n" + "="*80)
        print("SAMPLE NEGATIVE CONTEXTS EXAMPLES")
        print("="*80)
        
        examples_shown = 0
        for query_item in sample_results:
            if examples_shown >= 3:
                break
            
            if query_item.get('input_negative_ctxs'):
                print(f"Query {query_item['id']}:")
                print(f"  Question: {query_item.get('question', 'N/A')[:100]}...")
                print(f"  Negative docs extracted: {len(query_item['input_negative_ctxs'])}")
                for i, doc in enumerate(query_item['input_negative_ctxs'][:2], 1):
                    print(f"    Doc {i}: {doc['id']} (score: {doc.get('score', 'N/A')})")
                print()
                examples_shown += 1
    
    elif mode == "all":
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
            if examples_shown >= 5:
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
    MODE = "input"  # Change to "all" for original functionality, "input" for negative contexts extraction
    MAX_DOCS_PER_QUERY = 100  # For "all" mode
    NUM_NEGATIVE_DOCS = 3  # For "input" mode - number of top docs to extract as negative contexts
    
    if MODE == "all":
        # Original reranking mode
        retrieval_file = '/scratch/dq2024/diverse_retriever/inference/qwen3/base/qampari_corpus_finetuned_base_t500.json'
        gpt5_verifier_file = '/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/qwen3/qampari_base_qwen3-30B_all_gold_docs_500.json'
        #output_file = '/scratch/dq2024/diverse_retriever/eval/qwen3/qwen3_verifier/qampari_finetuned_base_t500_promote-qwen3.json'
        
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
        print_statistics(stats, reranked_results, gpt5_dict, max_docs=MAX_DOCS_PER_QUERY, mode=MODE)
        
        # Save results
        save_json_file(reranked_results, output_file)
        
        print(f"\nReranking complete!")
        print(f"Original queries: {len(retrieval_results)}")
        print(f"Reranked queries: {len(reranked_results)}")
        print(f"Documents per query in output: {MAX_DOCS_PER_QUERY}")
    
    elif MODE == "input":
        # Input negative contexts extraction mode
        retrieval_file = '/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/qwen3/qampari_base_qwen3-30B_all_gold_docs_200.json'
        output_file = '/scratch/dq2024/diverse_retriever/inference/qwen3_verifier/qwen3/input_files/qampari_finetuned_base_t200_promote-qwen3_0-6_unformat.json'
        
        NUM_NEGATIVE_DOCS = 6
        print(f"Loading retrieval file for negative contexts extraction...")
        retrieval_results = load_json_file(retrieval_file)
        
        print(f"Loaded {len(retrieval_results)} queries from retrieval results")
        
        # Process queries to extract input_negative_ctxs
        processed_results, stats = process_input_negative_mode(retrieval_results, num_negative_docs=NUM_NEGATIVE_DOCS)
        
        # Print statistics
        print_statistics(stats, processed_results, mode=MODE)
        
        # Save results
        save_json_file(processed_results, output_file, format_style="array")
        
        print(f"\nNegative contexts extraction complete!")
        print(f"Original queries: {len(retrieval_results)}")
        print(f"Processed queries: {len(processed_results)}")
        print(f"Negative docs per query: {NUM_NEGATIVE_DOCS}")


if __name__ == "__main__":
    main()