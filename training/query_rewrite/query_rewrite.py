import json
import torch
import multiprocessing as mp
from vllm import LLM, SamplingParams
from typing import List, Dict, Any, Optional, Tuple

# Fix CUDA multiprocessing issue
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

class QueryRewriterVLLM:
    def __init__(self, model_name, max_context_length):
        print(f"Loading vLLM model for query rewriting: {model_name}")
        
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=4, 
            gpu_memory_utilization=0.9, 
            max_model_len=max_context_length,
            trust_remote_code=True,
            dtype="auto"
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
    
    def create_rewrite_prompt(self, query: str, documents: List[Dict[str, Any]], 
                             rewrite_type: str = "comprehensive") -> str:
        """Create prompt for query rewriting based on gold documents"""
        
        # IMPORTANT: If no documents, return None to signal no rewriting needed
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
            prompt = f"""Your task is to extract keywords from the given documents and append them to the original query. You will be provided detailed instructions to complete this task.

    Original Query: {query}

    Documents:
    {documents_text}

    Instructions:
    1. Extract relevant keywords from the given documents
    2. Add the keywords to the end of the original query
    3. Do not add anything else besides the new keywords
    4. Your response should not modify the original query at all

    New Query:"""
            return prompt

    def rewrite_queries_batch(self, entries: List[Dict[str, Any]], 
                            rewrite_type: str = "comprehensive", 
                            batch_size: int = 8) -> List[str]:
        """Rewrite multiple queries in batches using vLLM"""
        rewritten_queries = []
        
        # Prepare all prompts
        all_prompts = []
        valid_indices = []
        no_docs_indices = []  # Track entries with no documents
        
        for i, entry in enumerate(entries):
            query = entry.get("question", "")
            documents = entry.get("ctxs", [])
            
            # Check if we have a valid query
            if not query:
                print(f"Warning: Entry {i+1} missing query", flush=True)
                rewritten_queries.append("")  # Empty for missing query
                continue
            
            # Check if we have documents
            if not documents:
                print(f"Info: Entry {i+1} has no documents - keeping original query", flush=True)
                no_docs_indices.append(i)
                rewritten_queries.append(query)  # Keep original query when no documents
            else:
                prompt = self.create_rewrite_prompt(query, documents, rewrite_type)
                if prompt is not None:  # Should always be true if documents exist
                    all_prompts.append(prompt)
                    valid_indices.append(i)
                else:
                    # Shouldn't happen, but handle it anyway
                    rewritten_queries.append(query)
        
        # Process in batches only for entries with documents
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
                    
                    # Store result at correct index
                    while len(rewritten_queries) <= orig_idx:
                        rewritten_queries.append("")
                    rewritten_queries[orig_idx] = rewritten
                    
                    original = entries[orig_idx].get("question", "")[:50]
                    print(f"  ✓ Rewritten query {orig_idx+1}: {original}...", flush=True)
                
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
        
        # Log summary of no-docs entries
        if no_docs_indices:
            print(f"\nTotal entries with no documents (kept original): {len(no_docs_indices)}", flush=True)
        
        return rewritten_queries

def process_multiple_entries(rewriter: QueryRewriterVLLM, 
                           input_file: str, 
                           output_file: str,
                           rewrite_type: str = "comprehensive",
                           batch_size: int = 8,
                           use_batch_processing: bool = True):
    """Process all JSON entries in a JSONL file with query rewriting"""
    
    processed_entries = []
    no_docs_count = 0  # Track entries with no documents
    
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
                    # Count entries with no documents
                    if not data.get("ctxs", []):
                        no_docs_count += 1
                except json.JSONDecodeError:
                    print(f"Error: Invalid JSON, skipping line", flush=True)
                    continue
        
        print(f"Loaded {len(all_entries)} entries for batch processing", flush=True)
        print(f"Entries with no documents: {no_docs_count}", flush=True)
        
        # Batch rewrite all queries
        rewritten_queries = rewriter.rewrite_queries_batch(
            all_entries, rewrite_type, batch_size
        )
        
        # Create new entries with rewritten queries
        for entry, rewritten in zip(all_entries, rewritten_queries):
            # Create a new simplified entry
            new_entry = {
                "question": rewritten,  # Rewritten query (or original if no docs)
                "original_query": entry.get("question", ""),  # Original query saved here
                "input": entry.get("input", ""),  # Keep input field if it exists
                "id": entry.get("id", 0),  # Keep id field
                "answers": entry.get("answers", []),  # Keep answers field if it exists
                "had_documents": len(entry.get("ctxs", [])) > 0  # Track if rewriting was attempted
            }
            
            # Only add input_negative_ctxs if it exists in the original
            if "input_negative_ctxs" in entry:
                new_entry["input_negative_ctxs"] = entry["input_negative_ctxs"]
            
            processed_entries.append(new_entry)
    
    else:
        # Process entries one by one (non-batch mode)
        with open(input_file, 'r', encoding='utf-8') as f:
            line_num = 0
            for line in f:
                line = line.strip()
                line_num += 1
                
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    print(f"\n--- Processing entry {line_num}/{total_lines} ---", flush=True)
                    print(f"Original query: {data.get('question', '')[:100]}...", flush=True)
                    
                    # Get rewritten query
                    query = data.get("question", "")
                    documents = data.get("ctxs", [])
                    
                    # Check if we have documents
                    if not documents:
                        print(f"No documents found - keeping original query", flush=True)
                        rewritten = query
                        no_docs_count += 1
                    elif query:
                        prompt = rewriter.create_rewrite_prompt(query, documents, rewrite_type)
                        if prompt is not None:
                            outputs = rewriter.llm.generate([prompt], rewriter.sampling_params)
                            rewritten = outputs[0].outputs[0].text.strip()
                            rewritten = rewritten.replace("Rewritten Query:", "").strip()
                            rewritten = rewritten.replace("Expanded Query:", "").strip()
                            rewritten = rewritten.replace("New Query:", "").strip()
                        else:
                            rewritten = query
                    else:
                        rewritten = query
                    
                    # Create new simplified entry
                    new_entry = {
                        "question": rewritten,
                        "original_query": query,
                        "input": data.get("input", ""),
                        "id": data.get("id", 0),
                        "answers": data.get("answers", []),
                        "had_documents": len(documents) > 0
                    }
                    
                    if "input_negative_ctxs" in data:
                        new_entry["input_negative_ctxs"] = data["input_negative_ctxs"]
                    
                    processed_entries.append(new_entry)
                    
                    if documents:
                        print(f"Rewritten query: {rewritten}", flush=True)
                    
                    # Optional: Save progress periodically
                    if line_num % 10 == 0:
                        temp_output = output_file.replace('.json', f'_temp_{line_num}.json')
                        with open(temp_output, 'w', encoding='utf-8') as temp_f:
                            for entry in processed_entries:
                                temp_f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                        print(f"Progress saved to {temp_output}", flush=True)
                    
                except json.JSONDecodeError:
                    print(f"Error: Invalid JSON on line {line_num}", flush=True)
                    continue
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}", flush=True)
                    continue
    
    # Save final results as JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in processed_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Print statistics
    print("\n=== Rewriting Statistics ===")
    print(f"Total entries processed: {len(processed_entries)}")
    print(f"Entries with no documents (kept original): {no_docs_count}")
    print(f"Entries with documents (rewritten): {len(processed_entries) - no_docs_count}")
    
    # Calculate average query length change only for entries that had documents
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
    # Configuration
    INPUT_FILE = "qwen3_base_finetuned_t200_MAX12.json"  # Your filtered gold documents file
    OUTPUT_FILE = "qampari_qwen3_rewritten_queries_and_keywords.json"
    MODEL_NAME = "Qwen/Qwen3-235B-A22B-Instruct-2507" 
    
    # Rewriting configuration
    REWRITE_TYPE = "keyword" 
    BATCH_SIZE = 16  
    USE_BATCH_PROCESSING = False  
    MAX_CONTEXT_LENGTH = 8096
    
    print("=== Query Rewriting with vLLM ===")
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
            sample_with_docs = [e for e in processed_entries if e.get("had_documents", False)][:2]
            sample_without_docs = [e for e in processed_entries if not e.get("had_documents", False)][:1]
            
            for i, entry in enumerate(sample_with_docs):
                print(f"\nExample {i+1} (had documents):")
                print(f"Original: {entry.get('original_query', '')[:150]}...")
                print(f"Rewritten: {entry.get('question', '')[:150]}...")
            
            for entry in sample_without_docs:
                print(f"\nExample (no documents - kept original):")
                print(f"Original: {entry.get('original_query', '')[:150]}...")
                print(f"Output: {entry.get('question', '')[:150]}...")
        
    except FileNotFoundError:
        print(f"Error: Could not find input file '{INPUT_FILE}'")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()