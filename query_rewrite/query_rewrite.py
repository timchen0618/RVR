import json
import torch
import multiprocessing as mp
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

from vllm import LLM, SamplingParams
from typing import List, Dict, Any, Optional, Tuple
import os

# Fix CUDA multiprocessing issue


class QueryRewriterVLLM:
    def __init__(self, model_name, max_context_length):
        print(f"Loading vLLM model for query rewriting: {model_name}")
        
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1, 
            gpu_memory_utilization=0.9, 
            max_model_len=max_context_length,
            trust_remote_code=True,
            dtype="auto",
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.7,  
            max_tokens=50,   
            top_p=0.8,
            top_k=20,
        )
        
        self.max_context_length = max_context_length
        
        print("vLLM model loaded successfully for query rewriting!", flush=True)
        print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
        print(f"Number of GPUs: {torch.cuda.device_count()}", flush=True)
    
    def create_rewrite_prompt(self, query: str, documents: List[Dict[str, Any]] = None, 
                             rewrite_type: str = "comprehensive", keywords: str = None) -> str:
        """Create prompt for query rewriting based on gold documents or keywords"""
        
        # Special handling for neg_keyword type
        if rewrite_type == "neg_keyword":
            if not keywords:
                return None
            
            prompt = f"""Your task is to rewrite the original query to find different information that is still related to the topic but explores alternative aspects NOT covered by the given keywords. 
            The new query will be used by a retrieval model to retrieve a new set of documents that captures new perspectives.

Original Query: {query}

Keywords to AVOID (these represent already found information): {keywords}

Instructions:
1. Create a new query that explores different aspects of the topic
2. The new query should not involve the given keywords
3. Maintain relevance to the original topic but seek alternative perspectives
4. DO NOT explicitly mention the keywords themselves
5. DO NOT repeat the prompt or the instructions in your answer
6. Frame the query about what you DO want to find
7. Your response SHOULD NOT have any new lines


New Query:"""
            return prompt
        
        # Original document-based rewriting cases
        if not documents:
            return None
        
        # Prepare document context
        doc_contexts = []
        for i, doc in enumerate(documents, 1):  
            title = doc.get('title', 'No title')
            text = doc.get('text', '')[:500]  
            score = doc.get('score', 'N/A')
            doc_contexts.append(f"Document {i} (Score: {score}):\nTitle: {title}\nText: {text}")
        
        documents_text = "\n\n".join(doc_contexts)
        
        if rewrite_type == "expansion":
            prompt = f"""Your task is to expand the original query with relevant terms and concepts from the documents. The new query will be used by a retrieval model to retrieve a new set of documents that
    captures new perspectives.

    Original Query: {query}

    Documents:
    {documents_text}

    Instructions:
    1. Create a new question using relevant terms and concepts from the documents
    2. DO NOT answer the question
    3. Your response should be one sentence
    4. Your response should end with a question mark

    New Query:"""
            return prompt
        elif rewrite_type == "keyword":
            prompt = f"""Your task is to append relevant terms from the documents to the original query. The original query and new terms will be used to retrieve a new set of documents that
            captures new perspectives. 

    Original Query: {query}

    Documents:
    {documents_text}

    Instructions:
    1. Gather relevant terms from the documents
    2. Create a new question by adding the relevant terms to the end of the query
    3. Stop responding after the relevant terms
    4. Do not repeat the prompt in your answer
    5. Your response should not have any new lines

    New Query:"""
            return prompt
        elif rewrite_type == "extract_keyword":
            prompt = f"""Your task is to gather a list of relevant terms from the documents that are related to the query. 

    Original Query: {query}

    Documents:
    {documents_text}

    Instructions:
    1. Gather a list relevant terms from the documents
    2. Only respond with a list of relevant terms
    3. Your response should not have incomplete words
    4. Do not repeat the prompt in your answer
    5. Your response should be on one line

    Relevant Terms:"""
            return prompt

    def rewrite_queries_batch(self, entries: List[Dict[str, Any]], 
                            rewrite_type: str = "comprehensive", 
                            batch_size: int = 8) -> List[str]:
        """Rewrite multiple queries in batches using vLLM"""
        rewritten_queries = []
        
        # Prepare all prompts
        all_prompts = []
        valid_indices = []
        skip_indices = []  # Track entries to skip
        
        for i, entry in enumerate(entries):
            query = entry.get("question_text", "")
            
            # Check if we have a valid query
            if not query:
                print(f"Warning: Entry {i+1} missing query", flush=True)
                rewritten_queries.append("")  # Empty for missing query
                continue
            
            # Handle neg_keyword type differently
            if rewrite_type == "neg_keyword":
                keywords = entry.get("keywords", "")
                if not keywords:
                    print(f"Info: Entry {i+1} has no keywords - keeping original query", flush=True)
                    skip_indices.append(i)
                    rewritten_queries.append(query)
                else:
                    prompt = self.create_rewrite_prompt(query, keywords=keywords, rewrite_type=rewrite_type)
                    if prompt is not None:
                        all_prompts.append(prompt)
                        valid_indices.append(i)
                    else:
                        rewritten_queries.append(query)
            else:
                # Original document-based logic
                documents = entry.get("ctxs", [])[:6]
                #documents = entry.get("ground_truths", [])[:]
                if not documents:
                    print(f"Info: Entry {i+1} has no documents - keeping original query", flush=True)
                    skip_indices.append(i)
                    rewritten_queries.append(query)
                else:
                    prompt = self.create_rewrite_prompt(query, documents, rewrite_type)
                    if prompt is not None:
                        all_prompts.append(prompt)
                        valid_indices.append(i)
                    else:
                        rewritten_queries.append(query)
        
        # Process in batches
        for i in range(0, len(all_prompts), batch_size):
            batch = all_prompts[i:i + batch_size]
            batch_indices = valid_indices[i:i + batch_size]
            
            print(f"Processing rewrite batch {i//batch_size + 1}: queries {i+1}-{min(i+batch_size, len(all_prompts))}", flush=True)
            
            try:
                # Generate rewritten queries for the batch
                outputs = self.llm.generate(batch, self.sampling_params)
                
                # Process responses
                for j, (output, orig_idx) in enumerate(zip(outputs, batch_indices)):
                    rewritten = output.outputs[0].text.strip()
                    
                    # Clean up the rewritten query
                    rewritten = rewritten.replace("Rewritten Query:", "").strip()
                    rewritten = rewritten.replace("Expanded Query:", "").strip()
                    rewritten = rewritten.replace("New Query:", "").strip()
                    rewritten = rewritten.replace("Relevant Terms:", "").strip()
                    rewritten = rewritten.split("\n")[0]
                    
                    # Store result at correct index
                    while len(rewritten_queries) <= orig_idx:
                        rewritten_queries.append("")
                    rewritten_queries[orig_idx] = rewritten
                    
                    original = entries[orig_idx].get("question", "")[:50]
                    print(f"  ✓ Rewritten query {orig_idx+1}: {original}...", flush=True)
                    print(f"rewritten: {rewritten}", flush=True)
                
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}", flush=True)
                # Keep original queries for failed batch
                for idx in batch_indices:
                    while len(rewritten_queries) <= idx:
                        rewritten_queries.append("")
                    rewritten_queries[idx] = entries[idx].get("question", "")
        
        # Fill in any missing entries with original queries
        for i, entry in enumerate(entries):
            if i >= len(rewritten_queries):
                rewritten_queries.append(entry.get("question", ""))
            elif not rewritten_queries[i]:
                rewritten_queries[i] = entry.get("question", "")
        
        # Log summary of skipped entries
        if skip_indices:
            skip_type = "keywords" if rewrite_type == "neg_keyword" else "documents"
            print(f"\nTotal entries with no {skip_type} (kept original): {len(skip_indices)}", flush=True)
        
        return rewritten_queries

def process_multiple_entries(rewriter: QueryRewriterVLLM, 
                           input_file: str, 
                           output_file: str,
                           rewrite_type: str = "comprehensive",
                           batch_size: int = 8,
                           use_batch_processing: bool = True):
    """Process all JSON entries in a JSONL file with query rewriting"""
    
    processed_entries = []
    skip_count = 0  # Track entries skipped
    
    # Count total lines for progress
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f if line.strip())
        print(f"Total entries to process: {total_lines}", flush=True)
    except:
        total_lines = "unknown"
    
    if use_batch_processing:
        # Load all entries first for batch processing
        all_entries = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    all_entries.append(data)
                    # Count entries to skip based on rewrite type
                    if rewrite_type == "neg_keyword":
                        if not data.get("keywords", ""):
                            skip_count += 1
                    elif rewrite_type == "extract_keyword":
                        if not data.get("ground_truths", ""):
                            skip_count += 1
                    else:
                        if not data.get("ctxs", []):
                            skip_count += 1
                except json.JSONDecodeError:
                    print(f"Error: Invalid JSON, skipping line", flush=True)
                    continue
        
        print(f"Loaded {len(all_entries)} entries for batch processing", flush=True)
        skip_type = "keywords" if rewrite_type == "neg_keyword" else "documents"
        print(f"Entries with no {skip_type}: {skip_count}", flush=True)
        
        # Batch rewrite all queries
        rewritten_queries = rewriter.rewrite_queries_batch(
            all_entries, rewrite_type, batch_size
        )
        
        # Create new entries with rewritten queries
        for entry, rewritten in zip(all_entries, rewritten_queries):
            # Create entry based on rewrite type
            if rewrite_type == "neg_keyword":
                new_entry = {
                    "question": rewritten,  # Rewritten query
                    "original_query": entry.get("question", ""),  # Original query
                    "avoided_keywords": entry.get("keywords", ""),  # Keywords that were avoided
                    "input": entry.get("input", ""),
                    "id": entry.get("id", 0),
                    "answers": entry.get("answers", []),
                    "had_keywords": bool(entry.get("keywords", ""))
                }
            elif rewrite_type == "extract_keyword":
                new_entry = {
                    "question": entry.get("question_text", ""),
                    "keywords": rewritten,
                    "input": entry.get("input", ""),
                    "id": entry.get("id", 0),
                    "answers": entry.get("answers", []),
                    "had_documents": len(entry.get("ground_truths", [])) > 0
                }
            else:
                new_entry = {
                    "question": rewritten,
                    "original_query": entry.get("question", ""),
                    "input": entry.get("input", ""),
                    "id": entry.get("id", 0),
                    "answers": entry.get("answers", []),
                    "had_documents": len(entry.get("ctxs", [])) > 0
                }
            
            # Only add input_negative_ctxs if it exists in the original
            if "input_negative_ctxs" in entry:
                new_entry["input_negative_ctxs"] = entry["input_negative_ctxs"]
            
            processed_entries.append(new_entry)
    
    if rewrite_type == "extract_keyword":
        for entry in processed_entries:
            if not entry.get("had_documents", False):
                entry["keywords"] = ""  # Clear keywords for entries without documents
        print(f"Cleared keywords for {sum(1 for e in processed_entries if not e.get('had_documents', False))} entries without documents")
    
    if rewrite_type == "neg_keyword":
        for entry in processed_entries:
            if not entry.get("had_keywords", False):
                entry["question"] = entry["original_query"]  # Clear keywords for entries without documents
        print(f"Cleared keywords for {sum(1 for e in processed_entries if not e.get('had_documents', False))} entries without documents")
    
    # Save final results as JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in processed_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Print statistics
    print("\n=== Rewriting Statistics ===")
    print(f"Total entries processed: {len(processed_entries)}")
    
    if rewrite_type == "neg_keyword":
        print(f"Entries with no keywords (kept original): {skip_count}")
        print(f"Entries with keywords (rewritten): {len(processed_entries) - skip_count}")
        
        # Calculate stats for neg_keyword rewriting
        entries_with_keywords = [e for e in processed_entries if e.get("had_keywords", False)]
        if entries_with_keywords:
            original_lengths = [len(e.get("original_query", "")) for e in entries_with_keywords]
            rewritten_lengths = [len(e.get("question", "")) for e in entries_with_keywords]
            avg_original = sum(original_lengths) / len(original_lengths) if original_lengths else 0
            avg_rewritten = sum(rewritten_lengths) / len(rewritten_lengths) if rewritten_lengths else 0
            
            print(f"\nFor entries with keywords to avoid:")
            print(f"  Average original query length: {avg_original:.1f} characters")
            print(f"  Average rewritten query length: {avg_rewritten:.1f} characters")
            print(f"  Average length change: {avg_rewritten - avg_original:+.1f} characters")
    else:
        print(f"Entries with no documents (kept original): {skip_count}")
        print(f"Entries with documents (rewritten): {len(processed_entries) - skip_count}")
        
        entries_with_docs = [e for e in processed_entries if e.get("had_documents", False)]
        if entries_with_docs:
            original_lengths = [len(e.get("original_query", "")) for e in entries_with_docs]
            rewritten_lengths = [len(e.get("question", "")) for e in entries_with_docs]
            avg_original = sum(original_lengths) / len(original_lengths) if original_lengths else 0
            avg_rewritten = sum(rewritten_lengths) / len(rewritten_lengths) if rewritten_lengths else 0
            
            print(f"\nFor entries with documents:")
            print(f"  Average original query length: {avg_original:.1f} characters")
            print(f"  Average rewritten query length: {avg_rewritten:.1f} characters")
            print(f"  Average length change: {avg_rewritten - avg_original:+.1f} characters")
    
    return processed_entries

def main():
    # Configuration for neg_keyword rewriting
    INPUT_FILE = "/scratch/dq2024/diverse_retriever/inference/contriever/base/all_golds/base_AR_gold_docs_200.json"  
    OUTPUT_FILE = "/contriever/contriever_keywords_max6.json"
    MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507" 
    
    # Rewriting configuration
    REWRITE_TYPE = "extract_keyword"  # New negation rewrite type
    BATCH_SIZE = 16  
    USE_BATCH_PROCESSING = True  
    MAX_CONTEXT_LENGTH = 8096
    
    print("=== Query Rewriting with Keyword Negation ===")
    print(f"Input file: {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Rewrite type: {REWRITE_TYPE}")
    print(f"Batch processing: {USE_BATCH_PROCESSING}")
    print(f"Batch size: {BATCH_SIZE}")
    print()
    
    # Initialize rewriter
    rewriter = QueryRewriterVLLM(MODEL_NAME, MAX_CONTEXT_LENGTH)
    
    try:
        # Process entries
        processed_entries = process_multiple_entries(
            rewriter, 
            INPUT_FILE, 
            OUTPUT_FILE,
            REWRITE_TYPE,
            BATCH_SIZE,
            USE_BATCH_PROCESSING
        )
        
        print(f"\n✓ Successfully processed {len(processed_entries)} entries")
        print(f"✓ Results saved to: {OUTPUT_FILE}")
        
        # Show sample results
        if processed_entries:
            print("\n=== Sample Results ===")
            sample_with_keywords = [e for e in processed_entries if e.get("had_keywords", False)][:2]
            sample_without_keywords = [e for e in processed_entries if not e.get("had_keywords", False)][:1]
            
            for i, entry in enumerate(sample_with_keywords):
                print(f"\nExample {i+1} (had keywords to avoid):")
                print(f"Original: {entry.get('original_query', '')[:150]}...")
                print(f"Keywords avoided: {entry.get('avoided_keywords', '')[:100]}...")
                print(f"Rewritten: {entry.get('question', '')[:150]}...")
            
            for entry in sample_without_keywords:
                print(f"\nExample (no keywords - kept original):")
                print(f"Query: {entry.get('question', '')[:150]}...")
        
    except FileNotFoundError:
        print(f"Error: Could not find input file '{INPUT_FILE}'")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()