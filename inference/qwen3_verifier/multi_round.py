#!/usr/bin/env python3
"""
Multi-round retrieval pipeline that processes documents through verification,
gold extraction, merging, and concatenation steps.

Pipeline Steps:
0. Clean Questions: Replace questions with raw queries from source file
1. Verify & Filter: Use Qwen3 to identify gold documents
2. Extract & Merge Golds: Keep running set of all gold documents across rounds
3. Concatenate Top N: Add top N golds to query
4. Merge with Initial: Combine all golds with current retrieval (rounds > 1 only)
"""

import json
import os
import sys
import argparse
import torch
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Set
from vllm import LLM, SamplingParams

# Fix CUDA multiprocessing issue
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)


class DocumentVerifierVLLM:
    """Document verifier using vLLM and Qwen3 model"""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"):
        print(f"Loading vLLM model: {model_name}", flush=True)
        
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            trust_remote_code=True,
            dtype="auto"
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=10,
            skip_special_tokens=True
        )
        
        print("vLLM model loaded successfully!", flush=True)
    
    def create_verification_prompt(self, query: str, document: Dict[str, Any]) -> str:
        """Create verification prompt for a document"""
        title = document['title'] if 'title' in document else 'No title'
        text = document['text'] if 'text' in document else ''
        
        return f"""You are a document relevance verifier. Your task is to determine if a given document contains a direct answer to a specific question.

Question: {query}

Document Title: {title}
Document Text: {text}

Instructions:
- Answer with ONLY "YES" if the document directly answers the question or contains clear, specific information that answers the question
- Answer with ONLY "NO" if the document does not contain a direct answer, is only tangentially related, or does not provide the specific information asked for
- Be strict in your evaluation - the document should contain factual information that directly addresses the query

Your answer (YES or NO):"""
    
    def verify_documents_batch(self, query: str, documents: List[Dict[str, Any]], 
                               batch_size: int = 16) -> List[bool]:
        """Verify multiple documents in batches using vLLM"""
        results = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            print(f"  Processing batch {i//batch_size + 1}: documents {i+1}-{min(i+batch_size, len(documents))}", 
                  flush=True)
            
            prompts = [self.create_verification_prompt(query, doc) for doc in batch]
            
            try:
                outputs = self.llm.generate(prompts, self.sampling_params)
                
                batch_results = []
                for j, output in enumerate(outputs):
                    response = output.outputs[0].text.strip().upper()
                    is_gold = "YES" in response and "NO" not in response
                    batch_results.append(is_gold)
                    
                    doc_title = batch[j]['title'] if 'title' in batch[j] else 'No title'
                    doc_title = doc_title[:50]
                    status = "✓ GOLD" if is_gold else "✗ FILTERED"
                    print(f"    {status} - Document {i+j+1}: {doc_title}...", flush=True)
                
                results.extend(batch_results)
                
            except Exception as e:
                print(f"  Error processing batch {i//batch_size + 1}: {e}", flush=True)
                results.extend([False] * len(batch))
        
        return results
    
    def filter_documents(self, data: Dict[str, Any], batch_size: int = 16) -> Dict[str, Any]:
        """Filter documents using batched processing"""
        query = data["question"] if "question" in data else ""
        documents = data["ctxs"] if "ctxs" in data else []
        
        if not query:
            print("  Warning: No question found in data", flush=True)
            return data
        
        if not documents:
            print("  Warning: No documents found in data", flush=True)
            return data
        
        print(f"  Query: {query[:100]}...", flush=True)
        print(f"  Evaluating {len(documents)} documents...", flush=True)
        
        verification_results = self.verify_documents_batch(query, documents, batch_size)
        
        gold_documents = [
            doc for doc, is_gold in zip(documents, verification_results) 
            if is_gold
        ]
        
        filtered_data = data.copy()
        filtered_data["ctxs"] = gold_documents
        
        print(f"  Original: {len(documents)} | Gold: {len(gold_documents)} | " +
              f"Filtered: {len(documents) - len(gold_documents)}", flush=True)
        
        return filtered_data


def extract_doc_ids(ctxs: List[Dict]) -> Set[str]:
    """Extract document IDs from contexts list"""
    return {ctx['id'] for ctx in ctxs if 'id' in ctx and ctx['id']}


def merge_contexts(retrieved_ctxs: List[Dict], ground_truth_ctxs: List[Dict], 
                   max_total: int = 100) -> List[Dict]:
    """
    Merge contexts by taking all ground truth docs and filling remaining slots
    with highest-scoring non-duplicate retrieved docs.
    """
    merged = ground_truth_ctxs.copy()
    used_ids = extract_doc_ids(ground_truth_ctxs)
    
    if len(merged) >= max_total:
        return merged[:max_total]
    
    sorted_retrieved = sorted(
        retrieved_ctxs,
        key=lambda x: float(x['score']) if 'score' in x else 0,
        reverse=True
    )
    
    for ctx in sorted_retrieved:
        if len(merged) >= max_total:
            break
        
        doc_id = ctx['id'] if 'id' in ctx else ''
        if doc_id and doc_id not in used_ids:
            merged.append(ctx)
            used_ids.add(doc_id)
    
    return merged


def concatenate_top_n_golds(query: str, gold_docs: List[Dict], top_n: int = 3) -> str:
    """
    Concatenate top N gold documents to the query.
    Format: "query [SEP] doc1_title: doc1_text [SEP] doc2_title: doc2_text ..."
    """
    # Sort by score (highest first)
    sorted_golds = sorted(gold_docs, key=lambda x: float(x['score']) if 'score' in x else 0, reverse=True)
    
    # Take top N
    top_golds = sorted_golds[:top_n]
    
    if not top_golds:
        return f"Question: {query}"
    
    # Build concatenated string matching concat.py format
    concatenated = f"Question: {query}\n\nDocuments: "
    
    for i, doc in enumerate(top_golds):
        title = doc['title'] if 'title' in doc else 'No title'
        text = doc['text'] if 'text' in doc else ''
        doc_text = f"{title} {text}"
        
        if i == 0:
            concatenated += doc_text
        else:
            concatenated += f"\n{doc_text}"
    
    return concatenated


def step0_clean_questions(source_file: str, 
                          target_file: str, 
                          output_file: str):
    """
    Step 0: Replace question fields in target file with questions from source file.
    This ensures we use raw queries for verification.
    Matches entries by index (line number).
    """
    print("\n" + "="*80)
    print("STEP 0: CLEAN QUESTIONS (Replace with raw queries)",flush=True)
    print("="*80)
    print(f"Source (raw questions): {source_file}",flush=True)
    print(f"Target (to update):     {target_file}",flush=True)
    print(f"Output:                 {output_file}",flush=True)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print("\nLoading source questions...",flush=True)
    source_questions = []
    with open(source_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    question = entry['question'] if 'question' in entry else (entry['question_text'] if 'question_text' in entry else '')
                    source_questions.append(question)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed line {line_num} in source file",flush=True)
                    source_questions.append(None)
    
    print(f"Loaded {len(source_questions)} questions from source file",flush=True)
    
    print("\nUpdating target file with raw questions...",flush=True)
    updated_entries = []
    with open(target_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    
                    # Get the corresponding question from source file (0-indexed)
                    idx = line_num - 1
                    if idx < len(source_questions) and source_questions[idx] is not None:
                        # Replace the question field
                        entry['question'] = source_questions[idx]
                        
                        # Also update 'input' field if it exists
                        if 'input' in entry:
                            entry['input'] = source_questions[idx]
                        
                        # Also update 'question_text' field if it exists
                        if 'question_text' in entry:
                            entry['question_text'] = source_questions[idx]
                    else:
                        print(f"Warning: No matching question for line {line_num}",flush=True)
                    
                    updated_entries.append(entry)
                    
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed line {line_num} in target file",flush=True)
                    continue
    
    print(f"Updated {len(updated_entries)} entries",flush=True)
    
    print("\nWriting cleaned file...",flush=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in updated_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\n✓ Step 0 complete: Questions replaced with raw queries",flush=True)
    print(f"✓ Saved to: {output_file}\n",flush=True)
    
    return updated_entries


def step1_verify_and_filter(verifier: DocumentVerifierVLLM, 
                            input_file: str, 
                            output_file: str,
                            batch_size: int = 16,
                            verify_top_k: int = 50):  # NEW PARAMETER
    """
    Step 1: Verify documents using Qwen3 and filter to keep only gold documents.
    Only verifies the top verify_top_k documents (by score).
    Saves to: round{#}_top{N}_filtered.json
    """
    print("\n" + "="*80)
    print("STEP 1: VERIFY & FILTER DOCUMENTS",flush=True)
    print("="*80)
    print(f"Input:  {input_file}",flush=True)
    print(f"Output: {output_file}",flush=True)
    print(f"Verification budget: Top {verify_top_k} documents",flush=True)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    processed_entries = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
        total_lines = len(lines)
    
    print(f"Total queries to process: {total_lines}\n",flush=True)
    
    for line_num, line in enumerate(lines, 1):
        try:
            data = json.loads(line)
            print(f"[{line_num}/{total_lines}] Processing query...", flush=True)
            
            # NEW: Only verify top K documents
            all_documents = data["ctxs"] if "ctxs" in data else []
            
            # Sort by score and take top K for verification
            sorted_docs = sorted(
                all_documents,
                key=lambda x: float(x['score']) if 'score' in x else 0,
                reverse=True
            )
            docs_to_verify = sorted_docs[:verify_top_k]
            
            print(f"  Total docs: {len(all_documents)} | Verifying top: {len(docs_to_verify)}",flush=True)
            
            # Create temporary data with only top K docs for verification
            temp_data = data.copy()
            temp_data["ctxs"] = docs_to_verify
            
            filtered_data = verifier.filter_documents(temp_data, batch_size)
            processed_entries.append(filtered_data)
            
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON on line {line_num}", flush=True)
            continue
        except Exception as e:
            print(f"Error processing line {line_num}: {e}", flush=True)
            continue
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in processed_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\n✓ Step 1 complete: {len(processed_entries)} queries processed",flush=True)
    print(f"✓ Saved to: {output_file}\n",flush=True)
    
    return processed_entries


def step2_extract_and_merge_golds(filtered_file: str,
                                   output_file: str,
                                   previous_golds_file: str = None):
    """
    Step 2: Extract all gold documents from filtered file.
    If not round 1, merge with previous round's gold documents.
    Saves to: round{#}_all_golds.json
    """
    print("\n" + "="*80)
    print("STEP 2: EXTRACT & MERGE GOLD DOCUMENTS",flush=True)
    print("="*80)
    print(f"Input:  {filtered_file}",flush=True)
    if previous_golds_file:
        print(f"Previous golds: {previous_golds_file}",flush=True)
    print(f"Output: {output_file}",flush=True)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load current filtered (gold) documents
    current_golds = []
    with open(filtered_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                current_golds.append(json.loads(line.strip()))
    
    # Load previous golds if this isn't round 1
    if previous_golds_file and os.path.exists(previous_golds_file):
        print(f"\nMerging with previous round's golds...",flush=True)
        previous_golds = []
        with open(previous_golds_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    previous_golds.append(json.loads(line.strip()))
        
        # Merge golds (remove duplicates per query)
        merged_golds = []
        for curr, prev in zip(current_golds, previous_golds):
            # Get IDs from both
            used_ids = extract_doc_ids(curr['ctxs'] if 'ctxs' in curr else [])
            prev_ctxs = prev['ctxs'] if 'ctxs' in prev else []
            
            # Start with current golds
            merged_ctxs = (curr['ctxs'] if 'ctxs' in curr else []).copy()
            
            # Add previous golds that aren't duplicates
            for doc in prev_ctxs:
                doc_id = doc['id'] if 'id' in doc else ''
                if doc_id and doc_id not in used_ids:
                    merged_ctxs.append(doc)
                    used_ids.add(doc_id)
            
            # Sort by score
            merged_ctxs.sort(key=lambda x: float(x['score']) if 'score' in x else 0, reverse=True)
            
            merged_entry = curr.copy()
            merged_entry['ctxs'] = merged_ctxs
            merged_golds.append(merged_entry)
        
        all_golds = merged_golds
        print(f"✓ Merged current and previous golds",flush=True)
    else:
        all_golds = current_golds
        print(f"✓ Using current golds only (round 1)",flush=True)
    
    # Save all golds
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in all_golds:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    total_gold_docs = sum(len(entry['ctxs']) if 'ctxs' in entry else 0 for entry in all_golds)
    print(f"\n✓ Step 2 complete: {len(all_golds)} queries processed",flush=True)
    print(f"✓ Total gold documents: {total_gold_docs}",flush=True)
    print(f"✓ Saved to: {output_file}\n",flush=True)
    
    return all_golds


def step3_concatenate_top_n(all_golds_file: str,
                            output_file: str,
                            top_n: int = 3):
    """
    Step 3: Select top N gold documents and concatenate them to the query.
    Saves to: round{#}_top{N}_concatenated.json
    """
    print("\n" + "="*80)
    print(f"STEP 3: CONCATENATE TOP {top_n} GOLDS TO QUERY")
    print("="*80)
    print(f"Input:  {all_golds_file}")
    print(f"Output: {output_file}")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load all golds
    all_golds = []
    with open(all_golds_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_golds.append(json.loads(line.strip()))
    
    # Concatenate top N golds to each query
    concatenated_entries = []
    for entry in all_golds:
        query = entry['question'] if 'question' in entry else ''
        gold_docs = entry['ctxs'] if 'ctxs' in entry else []
        
        # Create concatenated query
        concatenated_query = concatenate_top_n_golds(query, gold_docs, top_n)
        
        # Create new entry matching concat.py format exactly
        new_entry = {
            "question": concatenated_query,  # Concatenated version (matches concat.py)
            "input": concatenated_query,     # Same as question
            "id": entry['id'] if 'id' in entry else 0,
            "answers": entry['answers'] if 'answers' in entry else [],
            "ctxs": [],  # Empty like concat.py
            "input_negative_ctxs": []  # Empty like concat.py
        }
        
        concatenated_entries.append(new_entry)
    
    # Save as JSON array (not JSONL) with indent=4
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(concatenated_entries, f, indent=4, ensure_ascii=False)
    
    print(f"\n✓ Step 3 complete: {len(concatenated_entries)} queries processed")
    print(f"✓ Top {top_n} golds concatenated to each query")
    print(f"✓ Saved to: {output_file}\n")
    
    return concatenated_entries
    
    


def step4_merge_with_initial(all_golds_file: str,
                             initial_retrieval_file: str,
                             output_file: str,
                             max_total: int = 100):
    """
    Step 4: Merge all golds with initial retrieval file.
    Only runs if NOT round 1.
    Saves to: round{#}_merged.json
    """
    print("\n" + "="*80)
    print("STEP 4: MERGE GOLDS WITH INITIAL RETRIEVAL",flush=True)
    print("="*80)
    print(f"All golds: {all_golds_file}",flush=True)
    print(f"Initial retrieval: {initial_retrieval_file}",flush=True)
    print(f"Output: {output_file}",flush=True)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load both files
    all_golds = []
    with open(all_golds_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_golds.append(json.loads(line.strip()))
    
    initial_retrieval = []
    with open(initial_retrieval_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                initial_retrieval.append(json.loads(line.strip()))
    
    # Merge for each query
    merged_entries = []
    for gold_entry, initial_entry in zip(all_golds, initial_retrieval):
        merged_ctxs = merge_contexts(
            initial_entry['ctxs'] if 'ctxs' in initial_entry else [],
            gold_entry['ctxs'] if 'ctxs' in gold_entry else [],
            max_total=max_total
        )
        
        merged_entry = initial_entry.copy()
        merged_entry['ctxs'] = merged_ctxs
        merged_entries.append(merged_entry)
    
    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in merged_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\n✓ Step 4 complete: {len(merged_entries)} queries processed",flush=True)
    print(f"✓ Golds merged with initial retrieval (max {max_total} docs per query)",flush=True)
    print(f"✓ Saved to: {output_file}\n",flush=True)
    
    return merged_entries


def run_pipeline(args):
    """Run the complete multi-round pipeline"""
    
    print("\n" + "="*80)
    print("MULTI-ROUND RETRIEVAL PIPELINE",flush=True)
    print("="*80)
    print(f"Round: {args.round_num}",flush=True)
    print(f"Initial retrieval: {args.initial_retrieval_file}",flush=True)
    print(f"Source queries: {args.source_queries_file}",flush=True)
    print(f"Output base dir: {args.output_base_dir}",flush=True)
    print(f"Top N concatenate: {args.top_n_concat}",flush=True)
    print(f"Verifier: {args.verifier}",flush=True)
    print("="*80)
    
    # Define output paths
    base_dir = args.output_base_dir
    round_num = args.round_num
    top_n = args.top_n_concat
    
    # Step 0 output (cleaned file with raw questions)
    cleaned_file = os.path.join(base_dir, f"round{round_num}_cleaned.json")
    
    # Step 1 output
    filtered_dir = os.path.join(base_dir, "concatenated_queries")
    filtered_file = os.path.join(filtered_dir, f"round{round_num}_top{top_n}_filtered.json")
    
    # Step 2 output
    all_golds_dir = os.path.join(base_dir, "all_golds")
    all_golds_file = os.path.join(all_golds_dir, f"round{round_num}_all_golds.json")
    previous_golds_file = os.path.join(all_golds_dir, f"round{round_num-1}_all_golds.json") if round_num > 1 else None
    
    # Step 3 output
    verified_input_dir = os.path.join(base_dir, "verified_input")
    concatenated_file = os.path.join(verified_input_dir, f"round{round_num}_top{top_n}_concatenated.json")
    
    # Step 4 output
    merged_file = os.path.join(base_dir, f"round{round_num}_merged.json")
    
    # Step 0: Clean questions (replace with raw queries from source)
    step0_clean_questions(
        source_file=args.source_queries_file,
        target_file=args.initial_retrieval_file,
        output_file=cleaned_file
    )
    
    # Initialize verifier
    print("\nInitializing document verifier...",flush=True)
    verifier = DocumentVerifierVLLM(model_name=args.verifier)
    
    # # Step 1: Verify and filter (use cleaned file)
    # Step 1: Verify and filter (use cleaned file)
    step1_verify_and_filter(
        verifier=verifier,
        input_file=cleaned_file,
        output_file=filtered_file,
        batch_size=args.batch_size,
        verify_top_k=args.verify_top_k  # NEW
    )
    
    # Step 2: Extract and merge golds
    step2_extract_and_merge_golds(
        filtered_file=filtered_file,
        output_file=all_golds_file,
        previous_golds_file=previous_golds_file if round_num > 1 else None
    )
    
    # Step 3: Concatenate top N golds
    step3_concatenate_top_n(
        all_golds_file=filtered_file,
        output_file=concatenated_file,
        top_n=top_n
    )
    
    # Step 4: Merge with initial (only if round > 1)
    if round_num > 1:
        step4_merge_with_initial(
            all_golds_file=previous_golds_file,
            initial_retrieval_file=args.initial_retrieval_file,
            output_file=merged_file
        )
    else:
        print("\n" + "="*80)
        print("STEP 4: SKIPPED (Round 1)",flush=True)
        print("="*80)
        print("Step 4 only runs for rounds > 1\n",flush=True)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!",flush=True)
    print("="*80)
    print(f"\nOutput files:",flush=True)
    print(f"  0. Cleaned:      {cleaned_file}",flush=True)
    print(f"  1. Filtered:     {filtered_file}",flush=True)
    print(f"  2. All golds:    {all_golds_file}",flush=True)
    print(f"  3. Concatenated: {concatenated_file}",flush=True)
    if round_num > 1:
        print(f"  4. Merged:       {merged_file}",flush=True)
    print("\n",flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-round retrieval pipeline with verification and gold extraction"
    )
    
    parser.add_argument(
        "--initial_retrieval_file",
        type=str,
        required=True,
        help="Path to initial retrieval file (JSONL format)"
    )
    
    parser.add_argument(
        "--source_queries_file",
        type=str,
        required=True,
        help="Path to source file with raw queries (JSONL format)"
    )
    
    parser.add_argument(
        "--output_base_dir",
        type=str,
        required=True,
        help="Base directory for all output files"
    )
    
    parser.add_argument(
        "--round_num",
        type=int,
        required=True,
        help="Current round number (1, 2, 3, ...)"
    )
    
    parser.add_argument(
        "--top_n_concat",
        type=int,
        default=3,
        help="Number of top gold documents to concatenate to query (default: 3)"
    )
    
    parser.add_argument(
        "--verifier",
        type=str,
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="Model name for document verification (default: Qwen/Qwen3-30B-A3B-Instruct-2507)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for verification (default: 16)"
    )

    parser.add_argument(
        "--verify_top_k",
        type=int,
        default=50,
        help="Number of top-ranked documents to verify (default: 50)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.initial_retrieval_file):
        print(f"Error: Initial retrieval file not found: {args.initial_retrieval_file}",flush=True)
        sys.exit(1)
    
    if not os.path.exists(args.source_queries_file):
        print(f"Error: Source queries file not found: {args.source_queries_file}",flush=True)
        sys.exit(1)
    
    if args.round_num < 1:
        print(f"Error: Round number must be >= 1",flush=True)
        sys.exit(1)
    
    # Run pipeline
    run_pipeline(args)


if __name__ == "__main__":
    main()