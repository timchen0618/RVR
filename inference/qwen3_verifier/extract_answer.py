#!/usr/bin/env python3
"""
LLM-based Answer Extraction and Deduplication Pipeline

Takes verified documents and uses Qwen30B to:
1. Extract the answer string that each document provides for the query
2. Deduplicate documents based on LLM-extracted answers (not ground truth)
3. Select top N documents with unique LLM-extracted answers
4. Concatenate to create final input

This differs from the original pipeline which uses ground truth answers for deduplication.
Here, the LLM determines what answer each document is providing.
"""

import json
import os
import sys
import argparse
import torch
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Set, Optional
from vllm import LLM, SamplingParams
import time
from collections import defaultdict

# Fix CUDA multiprocessing issue
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)


class AnswerExtractorVLLM:
    """Extract answer strings from documents using vLLM and Qwen3 model"""
    
    def get_gpu_memory_mb(self):
        import subprocess
        """Get total GPU memory usage in MB using nvidia-smi"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                memory_values = [float(x) for x in result.stdout.strip().split('\n')]
                return memory_values[0]  # First GPU
        except Exception as e:
            print(f"Warning: Could not get GPU memory via nvidia-smi: {e}", flush=True)
        return 0.0
    
    def __init__(self, model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"):
        import subprocess
        print(f"Loading vLLM model: {model_name}", flush=True)
        self.total_docs_processed = 0
        self.total_processing_time = 0.0
        
        # Get GPU memory BEFORE loading (in MB)
        memory_before_mb = self.get_gpu_memory_mb()
        print(f"GPU memory before loading vLLM: {memory_before_mb:.0f} MB ({memory_before_mb/1000:.2f} GB)", flush=True)
        
        # Also check what's on the GPU
        try:
            result = subprocess.run(
                ['nvidia-smi'],
                stdout=subprocess.PIPE,
                text=True,
                timeout=5
            )
            print("\nGPU status before loading:")
            print(result.stdout)
        except:
            pass
        
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            trust_remote_code=True,
            dtype="auto"
        )
        
        # Wait for vLLM workers to fully initialize
        time.sleep(2)
        
        # Get GPU memory AFTER loading (in MB)
        memory_after_mb = self.get_gpu_memory_mb()
        self.llm_memory_gb = (memory_after_mb - memory_before_mb) / 1000
        
        print(f"GPU memory after loading vLLM: {memory_after_mb:.0f} MB ({memory_after_mb/1000:.2f} GB)", flush=True)
        print(f"✓ Answer Extractor LLM memory (delta): {self.llm_memory_gb:.2f} GB", flush=True)
        
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=20,  # Allow longer answers
            skip_special_tokens=True,
            stop=["\n", "\n\n", "Question:", "Short answer:", "Answer:", "The answer"]
        )
        
        print("vLLM model loaded successfully!", flush=True)

    def create_extraction_prompt(self, query: str, document: Dict[str, Any]) -> str:
        """Create prompt to extract the answer string from a document"""
        title = document.get('title', 'No title')
        text = document.get('text', '')
        
        return f"""You are an answer extraction assistant. Your task is to extract the specific answer to a question from a given document.

Question: {query}

Document Title: {title}
Document Text: {text}

Instructions:
- Extract ONLY the direct answer to the question from the document
- The answer should be a short phrase
- If the document contains multiple possible answers, extract the most prominent one
- Do not provide explanations, only the answer string

Answer:"""
#         return f"""Extract the short answer to the question from the document. Respond with ONLY the answer, nothing else. No explanation, no reasoning, no additional text.

# Question: {query}

# Document Title: {title}
# Document Text: {text}

# Answer:"""
    
    def extract_answers_batch(self, query: str, documents: List[Dict[str, Any]], 
                              batch_size: int = 16) -> List[str]:
        """Extract answer strings from multiple documents in batches"""
        results = []
        start_time = time.time()

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            print(f"  Processing batch {i//batch_size + 1}: documents {i+1}-{min(i+batch_size, len(documents))}", 
                  flush=True)
            
            prompts = [self.create_extraction_prompt(query, doc) for doc in batch]
            
            try:
                outputs = self.llm.generate(prompts, self.sampling_params)
                
                batch_results = []
                for j, output in enumerate(outputs):
                    answer_text = output.outputs[0].text.strip()
                    answer_text = answer_text.split('\n')[0].strip()
                    batch_results.append(answer_text)
                    
                    doc_title = batch[j].get('title', 'No title')[:50]
                    print(f"    Doc {i+j+1}: {doc_title}... → Answer: '{answer_text}'", flush=True)
                
                results.extend(batch_results)
                
            except Exception as e:
                print(f"  Error processing batch {i//batch_size + 1}: {e}", flush=True)
                # Return empty strings for failed batch
                results.extend([""] * len(batch))

        end_time = time.time()
        batch_time = end_time - start_time
        self.total_docs_processed += len(documents)
        self.total_processing_time += batch_time
        
        throughput = len(documents) / batch_time if batch_time > 0 else 0
        print(f"  Batch throughput: {throughput:.2f} docs/sec", flush=True)
        
        return results
    
    def print_efficiency_stats(self):
        """Print accumulated efficiency statistics"""
        avg_throughput = self.total_docs_processed / self.total_processing_time if self.total_processing_time > 0 else 0
        peak_memory = torch.cuda.max_memory_allocated() / 1e9  # GB
        
        print("\n" + "="*80)
        print("ANSWER EXTRACTOR EFFICIENCY STATS")
        print("="*80)
        print(f"Total documents processed: {self.total_docs_processed}")
        print(f"Total time: {self.total_processing_time:.2f}s")
        print(f"Average throughput: {avg_throughput:.2f} docs/sec")
        print(f"LLM memory: {self.llm_memory_gb:.2f} GB")
        print(f"Model: Qwen-30B (30B parameters)")
        print("="*80 + "\n", flush=True)


def normalize_answer(answer: str) -> str:
    """Normalize answer string for comparison"""
    # Remove extra whitespace
    answer = ' '.join(answer.split())
    # Convert to lowercase for comparison
    return answer.lower().strip()


def are_answers_similar(answer1: str, answer2: str, threshold: float = 0.8) -> bool:
    """
    Check if two answers are similar enough to be considered duplicates.
    Uses simple string similarity for now.
    """
    norm1 = normalize_answer(answer1)
    norm2 = normalize_answer(answer2)
    
    # Exact match after normalization
    if norm1 == norm2:
        return True
    
    # Check if one is a substring of the other (for partial matches)
    if norm1 in norm2 or norm2 in norm1:
        return True
    
    # Could add more sophisticated similarity metrics here (edit distance, etc.)
    
    return False


def deduplicate_by_llm_answers(documents: List[Dict[str, Any]], 
                                llm_answers: List[str],
                                top_n: int = 3) -> List[Dict[str, Any]]:
    """
    Deduplicate documents based on LLM-extracted answers.
    Keep top N documents with unique answers.
    
    Args:
        documents: List of document dicts (already sorted by score)
        llm_answers: List of answer strings extracted by LLM (parallel to documents)
        top_n: Number of unique documents to keep
    
    Returns:
        List of deduplicated documents
    """
    unique_docs = []
    seen_answers = []
    
    for doc, answer in zip(documents, llm_answers):
        # Skip documents with empty answers (extraction errors)
        if not answer or not answer.strip():
            print(f"    Skipping document (empty answer): {doc.get('title', 'No title')[:50]}...", flush=True)
            continue
        
        # Check if this answer is similar to any we've already seen
        is_duplicate = False
        for seen_answer in seen_answers:
            if are_answers_similar(answer, seen_answer):
                print(f"    Skipping document (duplicate answer '{answer}'): {doc.get('title', 'No title')[:50]}...", flush=True)
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_docs.append(doc)
            seen_answers.append(answer)
            print(f"    ✓ Keeping document (unique answer '{answer}'): {doc.get('title', 'No title')[:50]}...", flush=True)
        
        # Stop when we have enough unique documents
        if len(unique_docs) >= top_n:
            break
    
    return unique_docs


def concatenate_documents(query: str, documents: List[Dict], top_n: int = 3) -> str:
    """
    Concatenate documents to the query.
    Format: "Question: query\n\nDocuments: doc1_title: doc1_text\ndoc2_title: doc2_text ..."
    """
    if not documents:
        return f"Question: {query}"
    
    # Take only top N (should already be limited, but just in case)
    top_docs = documents[:top_n]
    
    # Build concatenated string
    concatenated = f"Question: {query}\n\nDocuments: "
    
    for i, doc in enumerate(top_docs):
        title = doc.get('title', 'No title')
        text = doc.get('text', '')
        doc_text = f"{title} {text}"
        
        if i == 0:
            concatenated += doc_text
        else:
            concatenated += f"\n{doc_text}"
    
    return concatenated


def process_file(extractor: AnswerExtractorVLLM,
                 input_file: str,
                 output_file: str,
                 top_n: int = 3,
                 batch_size: int = 16,
                 answers_only: bool = False,
                 negate_answers: bool = False):
    """
    Process a file of verified documents:
    1. Extract answer strings using LLM
    2. Deduplicate based on LLM answers
    3. Concatenate top N unique documents
    """
    print("\n" + "="*80)
    print("LLM ANSWER EXTRACTION & DEDUPLICATION")
    print("="*80)
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print(f"Top N:  {top_n}")
    print("="*80)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load input file
    entries = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line.strip()))
    
    print(f"\nLoaded {len(entries)} queries")
    
    # Process each query
    processed_entries = []
    stats = {
        'total_queries': len(entries),
        'total_docs_before': 0,
        'total_docs_after': 0,
        'queries_with_dedup': 0,
        'docs_with_empty_answer': 0
    }
    
    for idx, entry in enumerate(entries, 1):
        query = entry.get('question', '')
        documents = entry.get('ctxs', [])
        
        print(f"\n[{idx}/{len(entries)}] Processing query: {query[:100]}...")
        print(f"  Documents to process: {len(documents)}")
        
        if not documents:
            print("  Warning: No documents found, skipping")
            # Create entry with empty concatenation
            new_entry = {
                "question": f"Question: {query}",
                "input": f"Question: {query}",
                "id": entry.get('id', idx-1),
                "answers": entry.get('answers', []),
                "ctxs": [],
                "input_negative_ctxs": []
            }
            processed_entries.append(new_entry)
            continue
        
        # Sort documents by score (highest first)
        sorted_docs = sorted(
            documents,
            key=lambda x: float(x.get('score', 0)),
            reverse=True
        )
        
        # Extract answers from all documents
        print(f"  Extracting answers from {len(sorted_docs)} documents...")
        llm_answers = extractor.extract_answers_batch(query, sorted_docs[:top_n], batch_size)
        
        # Count docs with empty answers
        empty_answer_count = sum(1 for ans in llm_answers if not ans or not ans.strip())
        stats['docs_with_empty_answer'] += empty_answer_count
        
        # Deduplicate based on LLM answers
        if negate_answers:
            top_answers = []
            for ans in llm_answers[:top_n]:
                if ans and ans.strip():
                    top_answers.append(ans.strip())
            
            if top_answers:
                negation_list = " or ".join(top_answers)
                concatenated_query = (
                    f"Retrieve documents that answer the query, "
                    f"but do NOT retrieve documents about "
                    f"{negation_list}, "
                    f"as these answers are already covered.\n\n"
                    f"Query: {query}"
                )
            else:
                concatenated_query = f"Question: {query}"
            
            stats['total_docs_after'] += len(top_answers)
        elif answers_only:
            # Just take top N answers directly (no dedup)
            top_answers = []
            for ans in llm_answers[:top_n]:
                if ans and ans.strip():
                    top_answers.append(ans.strip())
            
            concatenated_query = f"Question: {query}\n\n" + "\n".join(top_answers)
            stats['total_docs_after'] += len(top_answers)
        else: 
            print(f"  Deduplicating based on LLM-extracted answers...")
            unique_docs = deduplicate_by_llm_answers(sorted_docs, llm_answers, top_n)
            
            # Update stats
            stats['total_docs_before'] += len(sorted_docs)
            stats['total_docs_after'] += len(unique_docs)
            if len(unique_docs) < len(sorted_docs):
                stats['queries_with_dedup'] += 1
            
            print(f"  Result: {len(sorted_docs)} docs → {len(unique_docs)} unique docs")
            
            # Concatenate unique documents
            concatenated_query = concatenate_documents(query, unique_docs, top_n)
        
        # Create output entry
        new_entry = {
            "question": concatenated_query,
            "input": concatenated_query,
            "id": entry.get('id', idx-1),
            "answers": entry.get('answers', []),
            "ctxs": [],
            "input_negative_ctxs": []
        }
        
        processed_entries.append(new_entry)
    
    # Save as JSON array (matching original format)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_entries, f, indent=4, ensure_ascii=False)
    
    # Print final stats
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"Total queries processed: {stats['total_queries']}")
    print(f"Total docs before dedup: {stats['total_docs_before']}")
    print(f"Total docs after dedup: {stats['total_docs_after']}")
    print(f"Queries with deduplication: {stats['queries_with_dedup']}")
    print(f"Docs with empty answers: {stats['docs_with_empty_answer']}")
    print(f"\n✓ Saved to: {output_file}")
    print("="*80 + "\n")
    
    extractor.print_efficiency_stats()


def main():
    parser = argparse.ArgumentParser(
        description="Extract answers from documents using LLM and deduplicate based on extracted answers"
    )
    
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input file with verified documents (JSONL format)"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output file for concatenated queries (JSON format)"
    )
    
    parser.add_argument(
        "--top_n",
        type=int,
        default=3,
        help="Number of unique documents to concatenate (default: 3)"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="Model name for answer extraction (default: Qwen/Qwen3-30B-A3B-Instruct-2507)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for LLM processing (default: 16)"
    )
    parser.add_argument(
        "--answers_only",
        action="store_true",
        default=False,
        help="If set, concatenate only LLM-extracted answers instead of full document text"
    )
    
    parser.add_argument(
        "--negate_answers",
        action="store_true",
        default=False,
        help="If set, create a negation prompt telling the retriever to avoid the extracted answers"
    )
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}", flush=True)
        sys.exit(1)
    
    # Initialize extractor
    print("\nInitializing answer extractor...")
    extractor = AnswerExtractorVLLM(model_name=args.model_name)
    
    # Process file
    process_file(
        extractor=extractor,
        input_file=args.input_file,
        output_file=args.output_file,
        top_n=args.top_n,
        batch_size=args.batch_size,
        answers_only=args.answers_only,
        negate_answers=args.negate_answers
    )


if __name__ == "__main__":
    main()